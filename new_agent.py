"""
This module implements a state-based graph workflow for processing financial transactions and user queries.
It uses a directed graph to process different types of inputs (transactions, OCR, questions) through various nodes.
"""

import datetime
from enum import Enum
from typing import List, TypedDict, Callable, Any
import uuid
from functools import wraps
import time
import logging
import os
import json

from postgre_db import TransactionCreate
from utils import DEFAULT_CURRENCY_ID, InputType, get_advisor_context_data, new_generate_ocr_table
from langgraph.graph import StateGraph, START, END


class ExtractedTransactionData(TypedDict):
    """Represents the output of a transaction processing operation.
    
    Attributes:
        category_id (str): Unique identifier for the transaction category
        category_name (str): Name of the transaction category
        user_id (str): Unique identifier of the user
        amount (float): Transaction amount
        date (str): Transaction date in YYYY-MM-DD format
        note (str | None): Optional note about the transaction
        source (str): Source of the transaction (e.g., 'text_input', 'ocr')
        image_url (str | None): Optional URL to an image associated with the transaction
    """
    category_id: str
    category_name: str
    user_id: str
    amount: float
    date: str
    note: str | None = None
    source: str
    user_id: str
    image_url: str | None = None
    currency_id: str
    
class UserInputState(TypedDict):
    """Base state for user input processing.
    
    Attributes:
        user_id (str): Unique identifier of the user
        user_input (str | None): Optional text input from user
        image_url (str | None): Optional URL to an image uploaded by user
    """
    user_id: str
    user_input: str | None = None
    image_url: str | None = None


class InputClassifierState(UserInputState):
    """State after input classification.
    
    Attributes:
        input_type (InputType): Type of input (TRANSACTION, OCR, or QUESTION)
    """
    input_type: InputType

class ExtractorState(InputClassifierState):
    """State after transaction extraction.
    
    Attributes:
        transaction_output (List[TransactionOutput]): List of extracted transactions
    """
    transaction_output: List[ExtractedTransactionData] | None = None

class DbInsertState(ExtractorState):
    """State after database insertion.
    
    Attributes:
        created_transaction_ids (List[str]): List of IDs of created transactions
    """
    created_transaction_ids: List[str]
    
class AdviceState(DbInsertState):
    """State after generating advice.
    
    Attributes:
        generated_advice (str): Generated financial advice
    """
    generated_advice: str
    generated_funny_response: str
    
class FunnyResponseState(AdviceState):
    """State after generating funny response.
    
    Attributes:
        generated_funny_response (str): Generated humorous response
    """
    generated_funny_response: str
    generated_advice: str
    

    
class QuestionState(InputClassifierState):
    """State for question processing.
    
    Attributes:
        user_context (str): Context of the user's question
        llm_response (str): Response from the language model
    """
    user_context: str
    llm_response: str

class OCRState(InputClassifierState):
    """State for OCR processing.
    
    Attributes:
        transaction_output (List[TransactionOutput]): List of extracted transactions
        ocr_text (str): Raw text extracted from the image
    """
    transaction_output: List[ExtractedTransactionData]
    ocr_text: str

def setup_logger(user_id: str):
    """Set up a logger that writes to a user-specific file.
    
    Args:
        user_id (str): User ID to use as the log file name
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create log file path
    log_file = f'logs/{user_id}.log'
    
    # Create logger
    logger = logging.getLogger(user_id)
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to prevent duplicate logging
    logger.handlers = []
    
    # Create file handler with append mode
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    
    # Create formatter with more detailed information
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    return logger

def create_middleware(func: Callable) -> Callable:
    """Creates a middleware function that wraps the original function.
    
    Args:
        func: The original node function to wrap
        
    Returns:
        A new function that includes timing, error handling, and state logging
    """
    @wraps(func)
    async def wrapper(state: Any) -> Any:
        # Get logger for this user
        logger = setup_logger(state["user_id"])
        
        start_time = time.time()
        logger.info(f"\n{'='*20} {func.__name__} {'='*20}")
        logger.info(f"Input state: {state}")
        
        try:
            result = await func(state)
            execution_time = time.time() - start_time
            logger.info(f"Execution time: {execution_time:.2f} seconds")
            logger.info(f"Output state: {result}")
            logger.info(f"{'='*50}\n")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(f"{'='*50}\n")
            raise
    
    return wrapper

# =============================NODES=============================

async def new_input_classifier_node(state: UserInputState) -> InputClassifierState:
    """Classifies the input type based on user input or image URL.
    
    Args:
        state (UserInputState): Current state containing user input
        
    Returns:
        InputClassifierState: New state with classified input type
    """
    from utils import classify_input_llm
    if state["image_url"] is not None:
        new_state = InputClassifierState(**state, input_type=InputType.OCR)
        return new_state
    else:
        input_type = await classify_input_llm(state["user_input"])
        new_state = InputClassifierState(**state, input_type=input_type)
        return new_state

async def new_ocr_node(state: InputClassifierState) -> OCRState:
    """Processes an image using OCR to extract transaction information.
    
    Args:
        state (InputClassifierState): Current state containing image URL
        
    Returns:
        OCRState: New state with extracted transaction information
    """
    transaction_create_list, ocr_text = await new_generate_ocr_table(state["image_url"], state["user_id"])
    new_state = OCRState(**state, transaction_output=transaction_create_list, ocr_text=ocr_text)
    return new_state

async def new_extractor_node(state: ExtractorState) -> ExtractorState:
    """Extracts transaction information from user input.
    
    Args:
        state (ExtractorState): Current state containing user input
        
    Returns:
        ExtractorState: New state with extracted transaction information
    """
    from utils import extract_user_input_info, classify_category_llm
    transaction = await extract_user_input_info(state["user_input"])
    category_data = await classify_category_llm(transaction["note"])
    category_id = category_data.split(":")[0]
    category_name = category_data.split(":")[1]
    transaction_output = {
        "category_id": category_id,
        "category_name": category_name,
        "amount": transaction["amount"],
        "date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "note": transaction["note"],
        "source": "text_input",
        "user_id": state["user_id"],
        "currency_id": DEFAULT_CURRENCY_ID,
        "image_url": None
    }
    new_state = ExtractorState(**state, transaction_output=[transaction_output])
    return new_state

async def new_insert_db_node(state: DbInsertState):
    """Inserts transactions into the database.
    
    Args:
        state (DbInsertState): Current state containing transactions to insert
        
    Returns:
        DbInsertState: New state with created transaction IDs
    """
    transaction_create_list = [
        TransactionCreate(
            userId=state["user_id"],
            amount=transaction["amount"],
            note=transaction["note"],
            date=transaction["date"],
            currencyId=transaction["currency_id"],
            categoryId=transaction["category_id"],
            imageUrl=transaction["image_url"]
        )
        for transaction in state["transaction_output"]
        ]
    
    transaction_ids = await insert_bulk_transactions_mock(transaction_create_list)
    new_state = DbInsertState(**state, created_transaction_ids=transaction_ids)
    return new_state

async def new_generate_funny_response_node(state: DbInsertState) -> FunnyResponseState:
    """Generates a humorous response based on the transaction.
    
    Args:
        state (DbInsertState): Current state containing transaction information
        
    Returns:
        FunnyResposeState: New state with generated funny response
    """
    from utils import generate_funny_response
    response = await generate_funny_response(state["user_input"])
    new_state = FunnyResponseState(**state, generated_funny_response=response)
    return new_state

async def new_generate_advice_node(state: DbInsertState) -> AdviceState:
    """Generates financial advice based on the transaction.
    
    Args:
        state (FunnyResposeState): Current state containing transaction information
        
    Returns:
        AdviceState: New state with generated advice
    """
    from utils import generate_advice_llm
    context = await get_advisor_context_data(state["user_id"], state["created_transaction_ids"])
    advice = await generate_advice_llm(context=context)
    new_state = AdviceState(**state, generated_advice=advice)
    return new_state
    
    # mock advice
    # advice = "MOCK ADVICE"
    # new_state = AdviceState(**state, generated_advice=advice)
    # return new_state

async def new_question_node(state: QuestionState) -> QuestionState:
    """Processes a user question and generates a response.
    
    Args:
        state (QuestionState): Current state containing user question
        
    Returns:
        QuestionState: New state with generated response
    """
    from utils import generate_rag, get_rag_context
    
    rag_context = await get_rag_context(state["user_id"])
    question = await generate_rag(rag_context, state["user_input"])
    new_state = QuestionState(**state, llm_response=question)
    return new_state

# =============================GRAPH=============================

class NodeName(str, Enum):
    """Enumeration of node names in the graph."""
    CLASSIFIER = "classifier"
    EXTRACTOR = "extractor"
    OCR = "ocr"
    QUESTION = "question"
    INSERT_DB = "insert_db"
    GENERATE_FUNNY_RESPONSE = "generate_funny_response"
    GENERATE_ADVICE = "generate_advice"

def get_classifier_next_node(x: InputClassifierState) -> str:
    """Determines the next node based on input type.
    
    Args:
        x (InputClassifierState): Current state with classified input type
        
    Returns:
        str: Name of the next node to process
    """
    if x["input_type"] == InputType.EXTRACTOR:
        return NodeName.EXTRACTOR.value
    elif x["input_type"] == InputType.OCR:
        return NodeName.OCR.value
    elif x["input_type"] == InputType.QUESTION:
        return NodeName.QUESTION.value
    else:
        raise ValueError(f"Invalid input type: {x['input_type']}")
    
def get_insert_db_next_node(x: DbInsertState) -> str:
    """Determines the next node based on input type.
    
    Args:
        x (DbInsertState): Current state with classified input type
        
    Returns:
        str: Name of the next node to process
    """
    if x["input_type"] == InputType.EXTRACTOR:
        return NodeName.GENERATE_ADVICE.value
    elif x["input_type"] == InputType.OCR:
        return END
    elif x["input_type"] == InputType.QUESTION:
        return END
    else:
        return END

# Initialize the graph
new_agent_graph = StateGraph(FunnyResponseState, input=UserInputState, output=FunnyResponseState)

# Add nodes to the graph with middleware
new_agent_graph.add_node(NodeName.CLASSIFIER.value, create_middleware(new_input_classifier_node))
new_agent_graph.add_node(NodeName.EXTRACTOR.value, create_middleware(new_extractor_node))
new_agent_graph.add_node(NodeName.INSERT_DB.value, create_middleware(new_insert_db_node))
new_agent_graph.add_node(NodeName.GENERATE_FUNNY_RESPONSE.value, create_middleware(new_generate_funny_response_node))
new_agent_graph.add_node(NodeName.GENERATE_ADVICE.value, create_middleware(new_generate_advice_node))
new_agent_graph.add_node(NodeName.QUESTION.value, create_middleware(new_question_node))
new_agent_graph.add_node(NodeName.OCR.value, create_middleware(new_ocr_node))

# Add edges to the graph
new_agent_graph.add_edge(START, NodeName.CLASSIFIER.value)

# Add conditional edges based on input type
new_agent_graph.add_conditional_edges(
    NodeName.CLASSIFIER.value,
    get_classifier_next_node,
)
new_agent_graph.add_edge(NodeName.EXTRACTOR.value, NodeName.INSERT_DB.value)
new_agent_graph.add_edge(NodeName.OCR.value, NodeName.INSERT_DB.value)

new_agent_graph.add_conditional_edges(
    NodeName.INSERT_DB.value,
    get_insert_db_next_node,
)

# Add remaining edges
new_agent_graph.add_edge(NodeName.GENERATE_ADVICE.value, NodeName.GENERATE_FUNNY_RESPONSE.value)
new_agent_graph.add_edge(NodeName.GENERATE_FUNNY_RESPONSE.value, END)
new_agent_graph.add_edge(NodeName.QUESTION.value, END)

# Compile the graph
new_agent_workflow = new_agent_graph.compile()

def get_graph():
    """Returns the graph structure in JSON format.
    
    Returns:
        str: JSON representation of the graph
    """
    graph = new_agent_workflow.get_graph()
    return graph.to_json()

# ---- MOCK DB ----

mock_db = []

async def insert_bulk_transactions_mock(transaction_create_list: List[TransactionCreate]) -> List[str]:
    """Mock function to simulate bulk transaction insertion.
    
    Args:
        transaction_create_list (List[TransactionCreate]): List of transactions to insert
        
    Returns:
        List[str]: List of generated transaction IDs
    """
    new_transaction_list = [
        {
           "id": str(uuid.uuid4()),
           "userId": transaction_create_list[t].userId,
           "amount": transaction_create_list[t].amount,
           "note": transaction_create_list[t].note,
           "date": transaction_create_list[t].date,
           "currencyId": transaction_create_list[t].currencyId,
           "categoryId": transaction_create_list[t].categoryId,
           "imageUrl": transaction_create_list[t].imageUrl
       }
       for t in range(len(transaction_create_list))
    ]
    mock_db.extend(new_transaction_list)    
    return [transaction["id"] for transaction in new_transaction_list]


# =============================INVOKE GRAPH=============================

async def invoke_graph_stream(initial_state: UserInputState):
    """Streams the output from each node in the graph.
    
    Args:
        initial_state (UserInputState): The initial state to start the graph processing
        
    Returns:
        AsyncIterator: An async iterator that yields streamed outputs
    """
    # Create an async iterator from the graph's astream
    stream = new_agent_workflow.astream(initial_state)
    # Track which responses have been yielded
    yielded_responses = set()
    
    async for chunk in stream:
        try:
            for node_name, node_output in chunk.items():
                print(f"Node name: {node_name}")
                if isinstance(node_output, dict):
                    if "transaction_output" in node_output and "transaction_output" not in yielded_responses:
                        transactions = node_output["transaction_output"]
                        if transactions:
                            yield json.dumps({
                                "type": "transactions",
                                "message": "Processing transactions...",
                                "data": transactions
                            })
                            yielded_responses.add("transaction_output")
                    
                    # Stream funny response
                    if "generated_funny_response" in node_output and "generated_funny_response" not in yielded_responses:
                        yield json.dumps({
                            "type": "funny_response",
                            "data": node_output["generated_funny_response"]
                        })
                        yielded_responses.add("generated_funny_response")

                    # Stream advice
                    if "generated_advice" in node_output and "generated_advice" not in yielded_responses:
                        print(f"Found advice: {node_output['generated_advice']}")  # Debug print
                        yield json.dumps({
                            "type": "advice",
                            "data": node_output["generated_advice"]
                        })
                        yielded_responses.add("generated_advice")

                    # Stream question response
                    if "llm_response" in node_output and "llm_response" not in yielded_responses:
                        yield json.dumps({
                            "type": "response",
                            "data": node_output["llm_response"]
                        })
                        yielded_responses.add("llm_response")
                    
                    # Break if we've reached the END node
                    if node_name == "__end__":
                        print("Reached END node")  # Debug print
                else:
                    print(f"Node output is not a dict: {node_output}")
        except Exception as e:
            print(f"Error in invoke_graph_stream: {e}")
            raise e