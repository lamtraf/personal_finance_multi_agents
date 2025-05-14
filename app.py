from fastapi import FastAPI, Header, Request, UploadFile, File, logger
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, List, TypedDict
import datetime

from langgraph.graph import StateGraph, START, END

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
from utils import generate_ocr_table

app = FastAPI()

# ====== INPUT & STATE ======
class FinanceInput(BaseModel):
    type: str
    content: str
    user_id: str
    
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
    subgraph_input = {"image_path": state["current_input"]["content"], "user_id": state["user_id"]}
    output = await ocr_subgraph.ainvoke(subgraph_input)
    state["transactions"].append(output["transaction"])
    return state

async def invoke_extractor(state: FinanceState) -> FinanceState:
    subgraph_input = {"text": state["current_input"]["content"], "user_id": state["user_id"]}
    output = await extractor_subgraph.ainvoke(subgraph_input)
    state["transactions"].extend(output["transactions"])
    return state

async def invoke_sentiment(state: FinanceState) -> FinanceState:
    subgraph_input = {"text": state["current_input"]["content"], "user_id": state["user_id"]}
    output = await sentiment_subgraph.ainvoke(subgraph_input)
    state["overall_sentiment"] = output["overall_sentiment"]
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
    print("USER ID DB INSERT:")
    print(state["user_id"])
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
workflow.add_edge("ocr", "sentiment")
workflow.add_edge("extractor", "sentiment")
workflow.add_edge("sentiment", "predictor")
workflow.add_edge("predictor", "db")
workflow.add_edge("db", "advisor")
workflow.add_edge("advisor", END)

graph = workflow.compile()

# ====== STREAMING RESPONSE ======
async def process_input(input_data: Dict, user_id: str):
    initial_state = {
        "messages": [],
        "current_input": input_data,
        "transactions": [],
        "overall_sentiment": "",
        "advice": "",
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
                        if "response" in t:
                            yield f"{t['transaction_id']}  || 🤣 Bot Says:\n{t['response']}\n"
                            response_printed = True
                            break
                if not advice_printed and "advice" in node_output:
                    # yield f"\n💡 Advice:\n{node_output['advice']}\n"
                    advice_printed = True
                
        if response_printed and advice_printed:
            break

# ====== ENDPOINTS ======
@app.post("/process_input")
async def process_user_input(finance_input: FinanceInput, request: Request = None):
    input_data = {"type": finance_input.type, "content": finance_input.content}
    return StreamingResponse(process_input(input_data, finance_input.user_id), media_type="text/plain")

@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    image_path = f"/tmp/{file.filename}"
    with open(image_path, "wb") as f:
        f.write(await file.read())
    return {"image_path": image_path}


@app.post("/process_input_sentiment")
async def process_input_sentiment(finance_input: FinanceInput):
    async def sentiment_streamer():
        subgraph_input = {
            "text": finance_input.content,
            "sentiment": "",
            "sentiment_score": 0.0,
            "response": ""
        }
        async for chunk in sentiment_subgraph.astream(subgraph_input):
            for node_output in chunk.values():
                if isinstance(node_output, dict):
                    response = node_output.get("response", "LLM không trả lời")
                    yield f"{response.strip()}\n"
                    return

        yield "⚠️ Không nhận được phản hồi.\n"

    return StreamingResponse(sentiment_streamer(), media_type="text/plain")


from langchain_core.runnables.graph import MermaidDrawMethod

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

@app.post("/process-ocr-new")
async def process_ocr_new(input: NewOCRInput):
    data = await generate_ocr_table(input.image_url, input.user_id)
    return data
