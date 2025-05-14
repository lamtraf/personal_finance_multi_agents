import asyncio
from langgraph.graph import StateGraph, START, END
from agents import ocr_subgraph, sentiment_subgraph, extractor_subgraph, predictor_subgraph, advisor_subgraph
from database import init_db, insert_transaction_pg, insert_prediction
from typing import TypedDict, List, Dict
import datetime

class FinanceState(TypedDict):
    messages: List[Dict]
    current_input: Dict
    transactions: List[Dict]
    overall_sentiment: str
    advice: str
    predictions: List[Dict]

async def invoke_ocr(state: FinanceState) -> FinanceState:
    subgraph_input = {"image_path": state["current_input"]["content"]}
    output = await ocr_subgraph.ainvoke(subgraph_input)
    state["transactions"].append(output["transaction"])  # ✅ support response
    return state

async def invoke_sentiment(state: FinanceState) -> FinanceState:
    subgraph_input = {"text": state["current_input"]["content"]}
    output = await sentiment_subgraph.ainvoke(subgraph_input)
    state["overall_sentiment"] = output["sentiment"]
    return state

async def invoke_extractor(state: FinanceState) -> FinanceState:
    subgraph_input = {"text": state["current_input"]["content"]}
    output = await extractor_subgraph.ainvoke(subgraph_input)
    state["transactions"].extend(output["transactions"])
    return state

async def invoke_predictor(state: FinanceState) -> FinanceState:
    subgraph_input = {"transactions": state["transactions"]}
    output = await predictor_subgraph.ainvoke(subgraph_input)
    state["predictions"] = output["predictions"]
    return state

async def invoke_advisor(state: FinanceState) -> FinanceState:
    subgraph_input = {
        "transactions": state["transactions"],
        "overall_sentiment": state["overall_sentiment"],
        "predictions": state["predictions"],
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

def generate_advice(state: FinanceState) -> str:
    total_spent = sum(t["amount"] for t in state["transactions"])
    predictions = state["predictions"]
    advice = f"Bạn đã chi {int(total_spent):,} VND trong tháng này. "
    if state["overall_sentiment"] == "lo lắng":
        advice += "Bạn đang lo lắng, hãy giảm chi tiêu không cần thiết. "
    if predictions:
        advice += f"Dự đoán chi tiêu tháng sau: {int(predictions[0]['predicted_amount']):,} VND cho {predictions[0]['category']} (độ tin cậy {predictions[0]['confidence']*100:.1f}%)."
    return advice

def router(state: FinanceState) -> str:
    if state["current_input"]["type"] == "image":
        return "ocr"
    elif state["current_input"]["type"] == "text":
        return "extractor"
    elif state["transactions"] and not state["overall_sentiment"]:
        return "process_sentiment"
    elif state["transactions"] and not state["predictions"]:
        return "predictor"
    elif state["predictions"]:
        return "db"
    return "end"

workflow = StateGraph(FinanceState)
workflow.add_node("ocr", invoke_ocr)
workflow.add_node("extractor", invoke_extractor)
workflow.add_node("process_sentiment", invoke_sentiment)
workflow.add_node("predictor", invoke_predictor)
workflow.add_node("db", db_insert_node)
workflow.add_node("advisor", invoke_advisor)

workflow.add_conditional_edges(START, router, {
    "ocr": "ocr",
    "extractor": "extractor",
    "process_sentiment": "process_sentiment",
    "predictor": "predictor",
    "db": "db",
    "end": "advisor"
})

workflow.add_edge("ocr", "process_sentiment")
workflow.add_edge("extractor", "process_sentiment")
workflow.add_edge("process_sentiment", "predictor")
workflow.add_edge("predictor", "db")
workflow.add_edge("db", "advisor")
workflow.add_edge("advisor", END)

graph = workflow.compile()

async def stream_response(inputs):
    config = {"configurable": {"thread_id": "2"}}
    initial_state = {
        "messages": [],
        "current_input": inputs,
        "transactions": [],
        "overall_sentiment": "",
        "advice": "",
        "predictions": []
    }

    response_printed = False
    advice_printed = False

    async for chunk in graph.astream(initial_state, config=config):
        for node_output in chunk.values():
            if isinstance(node_output, dict):
                if not response_printed:
                    for t in node_output.get("transactions", []):
                        if "response" in t:
                            print(t["response"])
                            response_printed = True
                            break  # dừng sau khi in 1 response

                if not advice_printed and "advice" in node_output:
                    print("\n💡 Final Advice:")
                    print(node_output["advice"])
                    advice_printed = True

        if response_printed and advice_printed:
            break  # kết thúc sớm nếu đã in đủ

                    
        yield chunk



async def run_example():
    init_db()
    inputs = {"type": "text", "content": "Mình vừa mua 300k tiền thịt heo ở VinMart"}
    async for _ in stream_response(inputs):
        pass

if __name__ == "__main__":
    asyncio.run(run_example())
