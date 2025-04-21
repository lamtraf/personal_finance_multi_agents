import requests
from functools import lru_cache
from config import LLAMA_CHAT_API_URL, LLAMA_GENERATE_API_URL, MODEL_NAME, CACHE_MAX_SIZE
import traceback

@lru_cache(maxsize=CACHE_MAX_SIZE)
def cached_llama_call(prompt: str) -> dict:
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    response = requests.post(LLAMA_GENERATE_API_URL, json=payload)
    return response.json()

async def ocr_process(image_path: str) -> tuple[str, dict]:
    text = f"Chi 1000000 cho thực phẩm tại VinMart, 2025-04-15"
    metadata = {"store": "VinMart", "location": "Hà Nội"}
    return text, metadata

async def analyze_sentiment_advanced(text: str) -> tuple[str, float]:
    prompt = f"Phân tích cảm xúc của văn bản sau: '{text}'"
    response = cached_llama_call(prompt)
    sentiment = "lo lắng" if "lo lắng" in text else "bình thường"
    score = 0.9 if "lo lắng" in text else 0.5
    return sentiment, score

async def extract_receipt_info_advanced(text: str) -> tuple[dict, dict]:
    parts = text.split(",")
    amount = float(parts[0].split()[-2])
    category = parts[0].split("cho")[-1].split("tại")[0].strip()
    date = parts[1].strip()
    metadata = {"store": parts[0].split("tại")[-1].strip(), "timestamp": "2025-04-15T10:00:00"}
    return {"date": date, "amount": amount, "category": category, "source": "biên lai"}, metadata

async def predict_spending(transactions: list[dict]) -> list[dict]:
    total = sum(t["amount"] for t in transactions)
    return [{"date": "2025-05-01", "predicted_amount": total * 1.1, "category": "thực phẩm", "confidence": 0.85}]


import re
import datetime
import httpx

async def extract_receipt_info_advanced(text: str):
    match = re.search(r'(\d+(?:[\.,]?\d+)?)(k|K|nghìn|tr|triệu)?', text)
    if not match:
        raise ValueError("Không tìm thấy số tiền trong nội dung.")

    num = match.group(1).replace(",", ".")
    unit = match.group(2) or ""
    amount = float(num)

    if unit.lower() in ['k', 'nghìn']:
        amount *= 1_000
    elif unit.lower() in ['tr', 'triệu']:
        amount *= 1_000_000

    transaction = {
        "amount": amount,
        "note": text,
    }
    metadata = {
        "source": "receipt",
        "raw_text": text
    }
    return transaction, metadata

# =============================
# 2. Rule-based fallback
# =============================
def classify_category(text: str) -> str:
    text = text.lower()

    categories = {
        "thực phẩm": ["gạo", "rau", "thịt", "trứng", "sữa", "ăn", "siêu thị", "vinmart"],
        "tiền nhà": ["thuê nhà", "tiền nhà", "phòng trọ", "nhà nguyên căn"],
        "tiền điện": ["tiền điện", "hóa đơn điện", "evn"],
        "tiền nước": ["tiền nước", "hóa đơn nước", "nước sinh hoạt"],
        "điện thoại / internet": ["viettel", "vina", "mobifone", "wifi", "data", "4g", "5g", "internet"],
        "đi lại": ["grab", "taxi", "xăng", "xe", "bus", "toll"],
        "giải trí": ["netflix", "karaoke", "xem phim", "rạp", "game", "spotify"],
        "mua sắm": ["quần áo", "giày", "mỹ phẩm", "shopee", "lazada", "tiki", "phụ kiện"],
        "giáo dục": ["học phí", "khóa học", "tiếng anh", "sách", "lớp học"],
        "y tế": ["thuốc", "khám", "bệnh viện", "hiệu thuốc", "y tế", "bảo hiểm y tế"],
    }

    for category, keywords in categories.items():
        if any(kw in text for kw in keywords):
            return category

    return "khác"

# =============================
# 3. LLM classification (Ollama /api/chat)
# =============================
async def classify_category_llm(text: str) -> str:
    prompt = f"""
Phân loại nội dung chi tiêu sau vào một trong các nhóm:
- thực phẩm
- tiền nhà
- tiền điện
- tiền nước
- điện thoại / internet
- đi lại
- giải trí
- mua sắm
- giáo dục
- y tế
- khác

Chỉ trả lời đúng một từ tên danh mục duy nhất.

Câu: "{text}"
"""

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            res = await client.post(
                LLAMA_CHAT_API_URL,
                json={
                    "model": "llama3.2",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                }
            )
            res.raise_for_status()
            data = res.json()
            return data["message"]["content"].strip().lower()
    except Exception:
        print("⚠️ Lỗi khi phân loại bằng LLM:")
        traceback.print_exc()
        return "khác"

# =============================
# 4. Tạo phản hồi hài hước (LLM via /api/chat)
# =============================
import httpx
import traceback

import httpx
import traceback

async def generate_funny_response(transaction: dict) -> str:
    prompt = f"""
Người dùng vừa chi {int(transaction['amount']):,} VND cho {transaction['category']} với ghi chú: "{transaction['note']}".
Viết một câu phản hồi hài hước, châm biếm, chỉ chửi người dùng nếu chi tiêu không hợp lý, tối đa 2 câu. Nhưng chỉ cần 1 câu là đủ.
"""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(
                LLAMA_CHAT_API_URL,
                json={
                    "model": "llama3.2",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                }
            )
            res.raise_for_status()
            data = res.json()
            response_text = data.get("message", {}).get("content", "").strip()
            if not response_text:
                return "🤖 Bot bí quá, chưa nghĩ ra câu nào hài!"
            return response_text

    except Exception:
        print("⚠️ Lỗi khi tạo câu hài hước:")
        traceback.print_exc()
        return "🤖 Bot đang lỗi kỹ thuật, không thể pha trò lúc này."



# =============================
# 5. Dummy support
# =============================

async def ocr_process(image_path: str):
    return "200k tiền gạo ở Bách Hóa Xanh", {"ocr_engine": "dummy", "image_path": image_path}



async def predict_spending(transactions: list):
    if not transactions:
        return []
    return [{
        "category": transactions[0]["category"],
        "predicted_amount": transactions[0]["amount"] * 1.2,
        "confidence": 0.87
    }]
    

