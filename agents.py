from fastapi import logger
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Dict
import logging

from config import LLAMA_CHAT_API_URL, LLAMA_GENERATE_API_URL
from postgre_db import create_transaction, insert_bulk_transactions

# ==== STATE DEFINITIONS ====

class OCRState(TypedDict):
    image_path: str
    extracted_text: str
    metadata: Dict
    user_id: str
    transactions: List[Dict]
    response: str
    overall_sentiment: str

class SentimentState(TypedDict):
    text: str
    sentiment: str
    sentiment_score: float
    response: str
    user_id: str
    transactions: List[Dict]

class ExtractorState(TypedDict):
    text: str
    transactions: List[Dict]
    user_id: str

class PredictorState(TypedDict):
    transactions: List[Dict]
    predictions: List[Dict]
    user_id: str
    
# ==== OCR WORKFLOW ====


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
    
async def ocr_node(state: OCRState) -> OCRState:
    from utils import generate_ocr_table, extract_receipt_info_advanced, generate_funny_response
    from database import insert_transaction_pg    
    transaction_create_list, ocr_text = await generate_ocr_table(state["image_path"], state["user_id"])
    try:
        transaction_ids = await insert_bulk_transactions(transaction_create_list)
        response = await generate_funny_response(ocr_text)
        
        enriched_transactions = []
        for transaction_id in transaction_ids:
            enriched_transactions.append({
                "transaction_id": transaction_id,
                "response": response
            })
        
        state["transactions"] = enriched_transactions
        state["response"] = response
        return state
    except Exception as e:
        raise e
    
ocr_workflow = StateGraph(OCRState)
ocr_workflow.add_node("process_ocr", ocr_node)
ocr_workflow.add_edge(START, "process_ocr")
ocr_workflow.add_edge("process_ocr", END)
ocr_subgraph = ocr_workflow.compile()

async def extractor_node(state: ExtractorState) -> ExtractorState:
    from utils import extract_receipt_info_advanced, classify_category_llm, generate_funny_response
    from database import insert_transaction_pg
    import datetime

    transaction, metadata = await extract_receipt_info_advanced(state["text"])
    note = transaction.get("note", "").lower().strip()

    category_with_id = await classify_category_llm(note)

    transaction["category"] = category_with_id.split(":")[1]
    transaction["category_id"] = category_with_id.split(":")[0]
    transaction.setdefault("date", datetime.datetime.now().strftime("%Y-%m-%d"))
    transaction.setdefault("source", "text_input")
    transaction.setdefault("user_id", state["user_id"])
    transaction_id = await insert_transaction_pg(transaction, metadata=metadata,)
    enriched = {**transaction, "metadata": metadata}
    try:
        response = await generate_funny_response(state["text"])
        enriched["response"] = response
        enriched["transaction_id"] = transaction_id
    except Exception as e:
        print(f"⚠️ Lỗi khi tạo câu hài hước: {e}")

    state["transactions"] = [enriched]
    return state


extractor_workflow = StateGraph(ExtractorState)
extractor_workflow.add_node("process_extractor", extractor_node)
extractor_workflow.add_edge(START, "process_extractor")
extractor_workflow.add_edge("process_extractor", END)

extractor_subgraph = extractor_workflow.compile()



# ==== PREDICTOR WORKFLOW ====

async def predictor_node(state: PredictorState) -> PredictorState:
    from utils import predict_spending
    print(state["transactions"])
    state["predictions"] = await predict_spending(state["transactions"])
    return state

predictor_workflow = StateGraph(PredictorState)
predictor_workflow.add_node("process_predictor", predictor_node)
predictor_workflow.add_edge(START, "process_predictor")
predictor_workflow.add_edge("process_predictor", END)
predictor_subgraph = predictor_workflow.compile()

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Dict

class AdvisorState(TypedDict):
    user_id: str
    transactions: List[Dict]
    overall_sentiment: str
    predictions: List[Dict]
    advice: str

async def advisor_node(state: AdvisorState) -> AdvisorState:
    from utils import generate_advice_llm, get_advisor_context_data
    context = await get_advisor_context_data(state["user_id"], state["transactions"])
    
    advice = await generate_advice_llm(context)
    state["advice"] = advice
    return state

advisor_workflow = StateGraph(AdvisorState)
advisor_workflow.add_node("generate_advice", advisor_node)
advisor_workflow.add_edge(START, "generate_advice")
advisor_workflow.add_edge("generate_advice", END)

advisor_subgraph = advisor_workflow.compile()

# ==== SENTIMENT WORKFLOW (Nâng cấp với Ollama) ====

from langgraph.graph import StateGraph, START, END


async def analyze_and_respond_node(state: SentimentState) -> SentimentState:
    import httpx
    import json
    
    logger.info(f"Sentiment State: {state}")
    
    transaction_ids = state["transactions"]

    prompt = f"""
    Phân tích cảm xúc của tin nhắn người dùng sau đây:

    "{state['text']}"

    Dựa trên cảm xúc của người dùng, trả lời ngắn gọn một câu phản hồi châm biếm dựa trên dựa trên cảm xúc đã phân tích được. 

    Trả lời theo định dạng JSON như sau:
    {{
        "sentiment": "positive" | "negative" | "worried",
        "sentiment_score": float (0 to 1),
        "response": "Ngắn gọn, tích cực hoặc hài hước"
    }}
    """
    
    print(f"Prompt: {prompt}")
    
    async with httpx.AsyncClient() as client:
        res = await client.post(
            LLAMA_CHAT_API_URL,
            json={
                "model": "llama3",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
        )
        try:
            raw = res.json()
            content = raw["message"]["content"]
            data = json.loads(content)
            state["sentiment"] = data.get("sentiment")
            state["sentiment_score"] = data.get("sentiment_score")
            state["explanation"] = data.get("explanation")
            state["response"] = data.get("response")
        except Exception as e:
            print(f"[Parse Error] {e}")
            state["response"] = "⚠️ Không thể phân tích hoặc tạo phản hồi lúc này."

    return state


sentiment_workflow = StateGraph(SentimentState)
sentiment_workflow.add_node("analyze_and_respond", analyze_and_respond_node)
sentiment_workflow.set_entry_point("analyze_and_respond")
sentiment_workflow.add_edge("analyze_and_respond", END)
sentiment_subgraph = sentiment_workflow.compile()
sentiment_subgraph = sentiment_workflow.compile()
