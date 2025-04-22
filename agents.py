from fastapi import logger
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Dict

from config import LLAMA_CHAT_API_URL
from postgre_db import create_transaction

# ==== STATE DEFINITIONS ====

class OCRState(TypedDict):
    image_path: str
    extracted_text: str
    transaction: Dict
    metadata: Dict
    user_id: str
    
class SentimentState(TypedDict):
    text: str
    sentiment: str
    sentiment_score: float
    user_id: str

class ExtractorState(TypedDict):
    text: str
    transactions: List[Dict]
    user_id: str

class PredictorState(TypedDict):
    transactions: List[Dict]
    predictions: List[Dict]
    user_id: str
    
# ==== OCR WORKFLOW ====

async def ocr_node(state: OCRState) -> OCRState:
    from utils import ocr_process, extract_receipt_info_advanced, generate_funny_response
    from database import insert_transaction
    import datetime

    state["extracted_text"], state["metadata"] = await ocr_process(state["image_path"])
    transaction, meta = await extract_receipt_info_advanced(state["extracted_text"])
    state["metadata"].update(meta)

    transaction.setdefault("category", "khác")
    transaction.setdefault("date", datetime.datetime.now().strftime("%Y-%m-%d"))
    transaction.setdefault("source", "ocr")
    transaction.setdefault("user_id", state["user_id"])
    insert_transaction(transaction, sentiment="không rõ", metadata=state["metadata"])

    enriched = {**transaction, "metadata": state["metadata"]}

    try:
        response = await generate_funny_response(transaction)
        enriched["response"] = response
        print(f"✅ OCR tạo câu hài hước: {response}")
    except Exception as e:
        print(f"⚠️ OCR lỗi khi tạo câu hài hước: {e}")

    state["transaction"] = enriched
    return state

ocr_workflow = StateGraph(OCRState)
ocr_workflow.add_node("process_ocr", ocr_node)
ocr_workflow.add_edge(START, "process_ocr")
ocr_workflow.add_edge("process_ocr", END)
ocr_subgraph = ocr_workflow.compile()

# ==== SENTIMENT WORKFLOW ====

# async def sentiment_node(state: SentimentState) -> SentimentState:
#     from utils import analyze_sentiment_advanced
#     state["sentiment"], state["sentiment_score"] = await analyze_sentiment_advanced(state["text"])
#     return state

# sentiment_workflow = StateGraph(SentimentState)
# sentiment_workflow.add_node("process_sentiment", sentiment_node)
# sentiment_workflow.add_edge(START, "process_sentiment")
# sentiment_workflow.add_edge("process_sentiment", END)
# sentiment_subgraph = sentiment_workflow.compile()

# ==== EXTRACTOR WORKFLOW ====

# Cache phân loại để không gọi LLM lại
category_cache = {}

async def extractor_node(state: ExtractorState) -> ExtractorState:
    from utils import extract_receipt_info_advanced, classify_category, classify_category_llm, generate_funny_response
    from database import insert_transaction
    import datetime

    transaction, metadata = await extract_receipt_info_advanced(state["text"])
    note = transaction.get("note", "").lower().strip()

    category_with_id = await classify_category_llm(note)

    transaction["category"] = category_with_id.split(":")[1]
    transaction["category_id"] = category_with_id.split(":")[0]
    transaction.setdefault("date", datetime.datetime.now().strftime("%Y-%m-%d"))
    transaction.setdefault("source", "text_input")
    transaction.setdefault("user_id", state["user_id"])
    await insert_transaction(transaction, sentiment="không rõ", metadata=metadata)

    enriched = {**transaction, "metadata": metadata}

    try:
        response = await generate_funny_response(transaction)
        enriched["response"] = response
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
    transactions: List[Dict]
    overall_sentiment: str
    predictions: List[Dict]
    advice: str

async def advisor_node(state: AdvisorState) -> AdvisorState:
    from utils import generate_advice_llm
    advice = await generate_advice_llm(state)
    state["advice"] = advice
    return state

advisor_workflow = StateGraph(AdvisorState)
advisor_workflow.add_node("generate_advice", advisor_node)
advisor_workflow.add_edge(START, "generate_advice")
advisor_workflow.add_edge("generate_advice", END)

advisor_subgraph = advisor_workflow.compile()

# ==== SENTIMENT WORKFLOW (Nâng cấp với Ollama) ====

from langgraph.graph import StateGraph, START, END

class SentimentState(TypedDict):
    text: str
    sentiment: str
    sentiment_score: float
    response: str
    user_id: str

async def analyze_and_respond_node(state: SentimentState) -> SentimentState:
    import httpx
    import json

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

    async with httpx.AsyncClient() as client:
        res = await client.post(
            LLAMA_CHAT_API_URL,
            json={
                "model": "llama3.2",
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


# Xây workflow mới chỉ với một node
sentiment_workflow = StateGraph(SentimentState)
sentiment_workflow.add_node("analyze_and_respond", analyze_and_respond_node)
sentiment_workflow.set_entry_point("analyze_and_respond")
sentiment_workflow.add_edge("analyze_and_respond", END)

sentiment_subgraph = sentiment_workflow.compile()

sentiment_subgraph = sentiment_workflow.compile()


# import asyncio
# from langgraph.graph import StateGraph


# async def test_workflow():
#     input_state = {
#         "text": "Nay tiêu hết 800k vào một bữa ăn rồi, buồn ghê.",
#         "sentiment": "",
#         "sentiment_score": 0.0,
#         "response": ""
#     }

#     async for chunk in sentiment_subgraph.astream(input_state):
#         for node_output in chunk.values():
#             print("\n🔹 Node output:")
#             for key, value in node_output.items():
#                 print(f"{key}: {value}")

# if __name__ == "__main__":
#     asyncio.run(test_workflow())

