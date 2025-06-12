from fastapi import FastAPI, Header, Request, UploadFile, File, logger
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, List, TypedDict
import logging
import datetime

from langgraph.graph import StateGraph, START, END

from new_agent import UserInputState, invoke_graph_stream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agents import (
    ocr_subgraph,
    extractor_subgraph,
    sentiment_subgraph,
    predictor_subgraph,
    advisor_subgraph
)
from agents import SentimentState, ExtractorState, PredictorState, AdvisorState
from database import insert_transaction_pg, insert_prediction
from postgre_db import get_categories
from utils import classify_input_llm, generate_ocr_table

app = FastAPI()

# ====== INPUT & STATE ======
class FinanceInput(BaseModel):
    type: str
    content: str
    user_id: str
    
class ClassifyInput(BaseModel):
    input: str

class NewOCRInput(BaseModel):
    image_url: str
    user_id: str

class FinanceState(TypedDict):
    user_id: str
    messages: List[Dict]
    current_input: Dict
    transactions: List[Dict]
    overall_sentiment: str
    advice: str
    predictions: List[Dict]
    user_id: str

# ====== NODES ======
async def invoke_ocr(state: FinanceState) -> FinanceState:
    subgraph_input = {"image_path": state["current_input"]["content"], "user_id": state["user_id"], "overall_sentiment": state["overall_sentiment"]}
    output = await ocr_subgraph.ainvoke(subgraph_input)
    return output


async def invoke_extractor(state: FinanceState) -> FinanceState:
    subgraph_input = {"text": state["current_input"]["content"], "user_id": state["user_id"]}
    output = await extractor_subgraph.ainvoke(subgraph_input)
    state["transactions"].extend(output["transactions"])
    return state

async def invoke_sentiment(state: FinanceState) -> FinanceState:
    subgraph_input = {"text": state["current_input"]["content"], "user_id": state["user_id"]}
    output = await sentiment_subgraph.ainvoke(subgraph_input)
    
    # state["overall_sentiment"] = output["sentiment"]
    return state

async def invoke_predictor(state: FinanceState) -> FinanceState:
    logger.info(f"TRANSACTIONS: {state['transactions']}")
    subgraph_input = {"transactions": state["transactions"], "user_id": state["user_id"]}
    output = await predictor_subgraph.ainvoke(subgraph_input)
    state["predictions"] = output["predictions"]
    logger.info(f"PREDICTIONS: {state['predictions']}")
    return state

async def invoke_advisor(state: FinanceState) -> FinanceState:
    subgraph_input = {
        "transactions": state["transactions"],
        "overall_sentiment": state["overall_sentiment"],
        "predictions": state["predictions"],
        "user_id": state["user_id"]
    }
    output = await advisor_subgraph.ainvoke(subgraph_input)
    state["advice"] = output["advice"]
    return state

async def db_insert_node(state: FinanceState) -> FinanceState:
    for t in state["transactions"]:
        t.setdefault("date", datetime.datetime.now().strftime("%Y-%m-%d"))
        t.setdefault("source", t.get("metadata", {}).get("source", "unknown"))
        insert_transaction_pg(t, state["overall_sentiment"], t.get("metadata", {}))
    for p in state["predictions"]:
        insert_prediction(p)
    return state

# ====== WORKFLOW ======
workflow = StateGraph(FinanceState)
workflow.add_node("ocr", invoke_ocr)
workflow.add_node("extractor", invoke_extractor)
workflow.add_node("sentiment", invoke_sentiment)
workflow.add_node("predictor", invoke_predictor)
workflow.add_node("db", db_insert_node)
workflow.add_node("advisor", invoke_advisor)

workflow.add_conditional_edges(
    START,
    lambda s: "ocr" if s["current_input"]["type"] == "image" else "extractor",
    {"ocr": "ocr", "extractor": "extractor"}
)

workflow.add_edge("ocr", END)
workflow.add_edge("extractor", "advisor")
workflow.add_edge("advisor", END)

graph = workflow.compile()

# ====== STREAMING RESPONSE ======
async def process_input(input_data: Dict, user_id: str):
    initial_state = {
        "messages": [],
        "current_input": input_data,
        "transactions": [],
        "overall_sentiment": "",
        "predictions": [],
        "user_id": user_id
    }
    response_printed = False
    advice_printed = False
    async for chunk in graph.astream(initial_state):
        for node_output in chunk.values():
            if isinstance(node_output, dict):
                if not response_printed:
                    for t in node_output.get("transactions", []):
                        transaction_ids = ",".join([t["transaction_id"] for t in node_output["transactions"]])
                        if "response" in t:
                            yield f"{transaction_ids}  || 🤣 Bot Says:\n{t['response']}\n"
                            response_printed = True
                            if (input_data["type"] == "image"):
                                logger.info(f"OCR TRANSACTIONS: {node_output['transactions']}")
                                advice_printed = True
                            break
                if not advice_printed and "advice" in node_output:
                    logger.info(f"ADVICE: {node_output['advice']}")
                    yield f"\n💡 Advice:\n{node_output['advice']}\n"
                    advice_printed = True
                
        if response_printed and advice_printed:
            break

# ====== ENDPOINTS ======
@app.post("/process_input")
async def process_user_input(finance_input: FinanceInput, request: Request = None):
    input_data = {"type": finance_input.type, "content": finance_input.content}
    return StreamingResponse(process_input(input_data, finance_input.user_id), media_type="text/plain")


@app.post("/process-ocr-langchain")
async def process_ocr_langchain(input: NewOCRInput):
    input_data = {"type": "image", "content": input.image_url}
    return StreamingResponse(process_input(input_data, input.user_id), media_type="text/plain")

@app.get("/graph")
async def get_graph():
    return {
        "graph": graph.get_graph().to_json(),
        "predictor": predictor_subgraph.get_graph().to_json(),
        "advisor": advisor_subgraph.get_graph().to_json(),
        "sentiment": sentiment_subgraph.get_graph().to_json(),
        "extractor": extractor_subgraph.get_graph().to_json(),
        "ocr": ocr_subgraph.get_graph().to_json()
    }

class APIUserInput(BaseModel):
    user_id: str
    user_input: str | None = None
    image_url: str | None = None
    
@app.post('/new-input')
async def new_input(input: APIUserInput):
    try:
        input_state = {
            "user_id": input.user_id,
            "user_input": input.user_input,
            "image_url": input.image_url
        }
        return StreamingResponse(
            invoke_graph_stream(input_state),
            media_type="text/plain",
            headers={
                "Transfer-Encoding": "chunked",
                "Connection": "keep-alive"
            }
        )
    except Exception as e:
        logger.error(f"Error in new-input endpoint: {str(e)}")
        raise