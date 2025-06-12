import datetime
import json
from typing import Dict, List
from fastapi import HTTPException
from PIL import Image
import requests
from functools import lru_cache
from config import GEMINI_API_KEY, LLAMA_CHAT_API_URL, LLAMA_GENERATE_API_URL, MODEL_NAME, CACHE_MAX_SIZE
import traceback

from database import insert_transaction_pg
from postgre_db import TransactionCreate, get_categories, insert_bulk_transactions

DEFAULT_CURRENCY_ID = "669d209b-99ac-401d-a441-8fa7bb387d4c"

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

async def extract_user_input_info(text: str) -> tuple[dict, dict]:
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

async def extract_user_input_info(text: str)-> Dict:
    match = re.search(r'(\d+(?:[\.,]?\d+)?)(k|K|nghìn|tr|triệu)?', text)

    if not match:
        raise HTTPException(status_code=400, detail="Không tìm thấy số tiền trong nội dung.")
    num = match.group(1).replace(",", "").replace(".", "")
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
    return transaction


GOOGLE_GEMINI_URL="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

async def classify_category_llm(text: str) -> tuple[str, str]:
    categories = await get_categories()
    categories_str = "\n".join([f"{cat.id}:{cat.name}" for cat in categories])
    prompt = f'Tôi có một danh sách các danh mục:\n{categories_str}.\nMỗi danh mục có 1 uuid và tên, cách nhau bởi dấu \":\".\nTôi sẽ cho bạn 1 câu nói, hãy phân loại nội dung câu nói đó vào một trong các nhóm danh mục trên.\nChỉ trả lời đúng danh mục duy nhất theo cú pháp "id:tên_danh_mục".\n\nSố tiền có thể có chứa đấu chấm (.) hoặc dấu phẩy (,) để phân cách phần nghìn. Hãy bỏ qua các kí tự này Ví dụ: 500.000 => 500000 hoặc 500,000 => 500000\n\nCâu: \"{text}\"'
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            res = await client.post(
                GOOGLE_GEMINI_URL,
                json={
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                },
                params={
                    "key": GEMINI_API_KEY
                }
            )
            res.raise_for_status()
            data = res.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip().lower()
            return text
    except Exception:
        print("⚠️ Lỗi khi phân loại bằng LLM:")
        traceback.print_exc()
        
        return "748a7d51-c8e4-48b6-8e8f-eb59bc341978:khác"

# =============================
# 4. Tạo phản hồi hài hước (LLM via /api/chat)
# =============================
import traceback

import httpx
import traceback

async def generate_funny_response(ocr_text) -> str:
    
    prompt = f"""
Người dùng vừa chi {ocr_text}
Viết một câu phản hồi hài hước, châm biếm, chỉ chửi người dùng nếu chi tiêu không hợp lý, tối đa 2 câu. Nhưng chỉ cần 1 câu là đủ. Chỉ trả lời bằng tiếng Việt. Không được đặt trong ngoặc kép
"""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(
                LLAMA_CHAT_API_URL,
                json={
                    "model": "llama3",
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
    
# extract table from receipt
async def generate_ocr_table(image_url: str, user_id: str) -> tuple[List[TransactionCreate], str]:
    from google import genai
    import requests
    from urllib.parse import urlparse
    import os
    import uuid

    # Validate URL
    parsed_url = urlparse(image_url)
    if parsed_url.scheme not in ['http', 'https']:
        raise ValueError("Only HTTP/HTTPS URLs are supported")

    # Generate unique filename with UUID
    file_extension = os.path.splitext(parsed_url.path)[1] or '.jpg'
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    
    # Download the image from URL
    response = requests.get(image_url)
    response.raise_for_status()
    
    # Save the image with UUID filename
    with open(unique_filename, 'wb') as f:
        f.write(response.content)
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        myfile = Image.open(unique_filename)
        transction_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[myfile, "Đây là hoá đơn mua hàng, hãy trích xuất thông tin của từng mặt hàng và tổng tiền của từng mặt hàng. theo cú pháp: <mặt_hàng>:<tổng_tiền>. Không cần đơn vị tiền tệ. Không trả lời thêm thông tin nào khác."])
        items = transction_response.text.split("\n")
        # extract data to a dict
        data = {}
        for item in items:
            if ":" in item:
                key, value = item.split(":")
                data[key.strip()] = int(value.strip().replace(',', '').replace('.', '').replace(' ', ''))
        if not data:
            raise HTTPException(status_code=400, detail="Không tìm thấy dữ liệu trong hóa đơn")
        
        categories = await get_categories()
        categories_str = "\n".join([f"{cat.id}:{cat.name}" for cat in categories])
        joined_list = "\n".join([f"\"{key} : {value}\"" for key, value in data.items()])
        category_prompt = f'Tôi có một danh sách các danh mục:\n{categories_str}.\nMỗi danh mục có 1 uuid và tên, cách nhau bởi dấu \":\".\nTôi sẽ cho bạn danh sách câu nói sau:\n\n{joined_list}\n\nHãy phân loại nội dung từng câu vào một trong các nhóm danh mục trên. Chỉ trả lời danh sách theo cú pháp: <id_danh_mục:số_tiền> phân cách bởi dấu phẩy và không trả lời thêm thông tin gì khác. Nếu không tìm thấy danh mục phù hợp thì trả về id "khác". Loại bỏ các kí tự xuống dòng. Số tiền có thể có chứa đấu chấm (.) hoặc dấu phẩy (,) để phân cách phần nghìn. Hãy bỏ qua các kí tự này Ví dụ: 500.000 => 500000 hoặc 500,000 => 500000'      
        category_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[category_prompt])        
        category_list = [
            TransactionCreate(
                userId=user_id,
                amount=int(item.split(":")[1].strip().replace(',', '').replace('.', '')),
                categoryId=item.split(":")[0].strip(),
                note=item.split(":")[1].strip(),
                currencyId=DEFAULT_CURRENCY_ID,
                imageUrl=image_url,
            )
            for item in category_response.text.split(",")
        ]
        return category_list, joined_list
    finally:
        # Clean up the downloaded file
        if os.path.exists(unique_filename):
            os.remove(unique_filename)
 
async def get_advisor_context_data(user_id: str, new_created_transaction_ids: list[str]) -> str:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://192.168.0.109:3000/api/v1/chat/advisor-context",
                json={
                    "userId": user_id,
                    "newCreatedTransactionIds": new_created_transaction_ids
                }
            )
            response.raise_for_status()
            context = response.json().get("context")
            return context
            
    except Exception as e:
        print("⚠️ Lỗi khi lấy dữ liệu ngữ cảnh:")
        traceback.print_exc()
        return "Không thể lấy dữ liệu ngữ cảnh lúc này."

async def generate_advice_llm(context: Dict) -> str:
    new_created_transactions = context.get("newTransactions")
    profile = context.get("context")
    prompt = f"""
    Tôi có một ngữ cảnh về tài chính của người dùng dạng JSON như sau:
    {profile}
    
    Người dùng vừa thêm mới một số giao dịch:
    {new_created_transactions}
    
    Hãy dựa vào thông tin trên. Hãy đưa ra câu trả lời theo 1 trong các mẫu sau:
    + Nếu người dùng cần hạn chế chi tiêu thêm, hãy đưa ra các gợi ý về việc hạn chế chi tiêu
    + Nếu người dùng cần tăng chi tiêu, hãy đưa ra các gợi ý về việc tăng chi tiêu
    + Nếu người dùng cần tăng thu nhập, hãy đưa ra các gợi ý về việc tăng thu nhập
    + Nếu người dùng cần giảm chi tiêu, hãy đưa ra các gợi ý về việc giảm chi tiêu
    + Nếu người dùng cần tăng thu nhập, hãy đưa ra các gợi ý về việc tăng thu nhập
    Nếu không thể trả lời được, thì trả lời "Không có lời khuyên"
    Chỉ sử dụng tiếng việt, không quá 100 từ
    """
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(
                LLAMA_CHAT_API_URL,
                json={
                    "model": "llama3",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                }
            )
            res.raise_for_status()
            data = res.json()
            response_text = data.get("message", {}).get("content", "").strip()
            return response_text
    except Exception:
        print("⚠️ Lỗi khi tạo câu hướng dẫn:")
        traceback.print_exc()
        return "🤖 Bot đang lỗi kỹ thuật, không thể tạo câu hướng dẫn lúc này."

from enum import Enum

class InputType(Enum):
    QUESTION = "question"
    EXTRACTOR = "extractor"
    OCR = "ocr"
    ERROR = "error"

async def classify_input_llm(text: str) -> InputType:
    prompt = f"""
    Tôi có một câu nói, hãy phân loại câu nói đó vào một trong các nhóm sau:
    + "{InputType.EXTRACTOR.value}" nếu câu nói là về giao dịch
    + "{InputType.QUESTION.value}" nếu câu nói là về câu hỏi
    + "{InputType.ERROR.value}" nếu câu nói không phải là về giao dịch hoặc câu hỏi
    chỉ trả lời "{InputType.EXTRACTOR.value}" hoặc "{InputType.QUESTION.value}" hoặc "{InputType.ERROR.value}", dạng lowercase
    Câu nói: {text}
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(
                LLAMA_CHAT_API_URL,
                json={
                    "model": "llama3",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                }
            )
            res.raise_for_status()
            data = res.json()
            response_text = data.get("message", {}).get("content", "").strip()
            return InputType(response_text)
    except Exception:
        print("⚠️ Lỗi khi phân loại câu nói:")
        traceback.print_exc()
        return "🤖 Bot đang lỗi kỹ thuật, không thể phân loại câu nói lúc này."


# =============================REFACTORED CODE=============================


# extract table from receipt
async def new_generate_ocr_table(image_url: str, user_id: str) -> tuple[List[TransactionCreate], str]:
    from google import genai
    import requests
    from urllib.parse import urlparse
    import os
    import uuid

    # Validate URL
    parsed_url = urlparse(image_url)
    if parsed_url.scheme not in ['http', 'https']:
        raise ValueError("Only HTTP/HTTPS URLs are supported")

    # Generate unique filename with UUID
    file_extension = os.path.splitext(parsed_url.path)[1] or '.jpg'
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    
    # Download the image from URL
    response = requests.get(image_url)
    response.raise_for_status()
    
    # Save the image with UUID filename
    with open(unique_filename, 'wb') as f:
        f.write(response.content)
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        myfile = Image.open(unique_filename)
        transction_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[myfile, "Đây là hoá đơn mua hàng, hãy trích xuất thông tin của từng mặt hàng và tổng tiền của từng mặt hàng. theo cú pháp: <mặt_hàng>:<tổng_tiền>. Không cần đơn vị tiền tệ. Không trả lời thêm thông tin nào khác."])
        items = transction_response.text.split("\n")
        # extract data to a dict
        data = {}
        for item in items:
            if ":" in item:
                key, value = item.split(":")
                data[key.strip()] = int(value.strip().replace(',', '').replace('.', '').replace(' ', ''))
        if not data:
            raise HTTPException(status_code=400, detail="Không tìm thấy dữ liệu trong hóa đơn")
        
        categories = await get_categories()
        categories_str = "\n".join([f"{cat.id}:{cat.name}" for cat in categories])
        joined_list = "\n".join([f"\"{key} : {value}\"" for key, value in data.items()])
        category_prompt = f'Tôi có một danh sách các danh mục:\n{categories_str}.\nMỗi danh mục có 1 uuid và tên, cách nhau bởi dấu \":\".\nTôi sẽ cho bạn danh sách câu nói sau:\n\n{joined_list}\n\nHãy phân loại nội dung từng câu vào một trong các nhóm danh mục trên. Chỉ trả lời danh sách theo cú pháp: <id_danh_mục:số_tiền> phân cách bởi dấu phẩy và không trả lời thêm thông tin gì khác. Nếu không tìm thấy danh mục phù hợp thì trả về id "khác". Loại bỏ các kí tự xuống dòng. Số tiền có thể có chứa đấu chấm (.) hoặc dấu phẩy (,) để phân cách phần nghìn. Hãy bỏ qua các kí tự này Ví dụ: 500.000 => 500000 hoặc 500,000 => 500000'      
        category_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[category_prompt])        
        category_list = [
            {
                "category_id":item.split(":")[0].strip(),
                "category_name":item.split(":")[1].strip(),
                "amount":int(item.split(":")[1].strip().replace(',', '').replace('.', '')),
                "date":datetime.datetime.now().strftime("%Y-%m-%d"),
                "note":item.split(":")[1].strip(),
                "source":"text_input",
                "user_id":user_id,
                "image_url":image_url,
                "currency_id": DEFAULT_CURRENCY_ID
            }
            
            for item in category_response.text.split(",")
        ]
        return category_list, joined_list
    finally:
        # Clean up the downloaded file
        if os.path.exists(unique_filename):
            os.remove(unique_filename)
            
async def get_rag_context(user_id: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://192.168.0.109:3000/api/v1/chat/rag-context",
                json={"userId": user_id}
            )   
            response.raise_for_status()
            context = response.json().get("context")
            return context
    except Exception as e:
        print("⚠️ Lỗi khi lấy dữ liệu ngữ cảnh:")
        traceback.print_exc()
        raise e
    

    """
    private formatPrompt(question: string, context: any): string {
        const userCurrency = context.userProfile?.currency as UserCurrency;
        const currencyInfo = userCurrency
            ? `Đơn vị tiền tệ của người dùng là ${userCurrency.name} (${userCurrency.symbol}, ${userCurrency.code}).`
            : 'Đơn vị tiền tệ của người dùng chưa được xác định.';

        const currencySymbol = userCurrency?.symbol || '₫';

        return `
Bạn là một trợ lý tài chính giúp người dùng hiểu về chi tiêu của họ.
${currencyInfo}
Vui lòng cung cấp câu trả lời tự nhiên dựa trên dữ liệu sau:

Câu hỏi: ${question}

Dữ liệu:
${JSON.stringify(context)}

Vui lòng cung cấp câu trả lời hữu ích và tự nhiên:
1. Trả lời trực tiếp câu hỏi của người dùng. Không được trả lời các nội dung không liên quan tới câu hỏi
2. Sử dụng ký hiệu tiền tệ chính xác (${currencySymbol}) khi đề cập đến số tiền
3. Sử dụng giọng điệu thân thiện, gần gũi
4. Sử dụng tiếng Việt
5. Không đưa ra câu hỏi của người dùng vào câu trả lời
6. Nếu không thể trả lời câu hỏi, hoặc câu hỏi không liên quan tới quản lí tài chính của người dùng, hãy trả lời "Không thể trả lời câu hỏi"
Trả lời:`;
    }
    """
    
def format_rag_prompt(question: str, context: Dict) -> str:
    user_currency = context.get("userProfile").get("currency")
    currency_info = f"Đơn vị tiền tệ của người dùng là {user_currency.get('name')} ({user_currency.get('symbol')}, {user_currency.get('code')})."
    currency_symbol = user_currency.get("symbol") or "₫"
    return f"""
    Bạn là một trợ lý tài chính giúp người dùng hiểu về chi tiêu của họ.
    {currency_info}
    Vui lòng cung cấp câu trả lời tự nhiên dựa trên dữ liệu sau:
    Câu hỏi: {question}
    Dữ liệu: {json.dumps(context)}
    Vui lòng cung cấp câu trả lời hữu ích và tự nhiên:
1. Trả lời trực tiếp câu hỏi của người dùng. Không được trả lời các nội dung không liên quan tới câu hỏi
2. Sử dụng ký hiệu tiền tệ chính xác ({currency_symbol}) khi đề cập đến số tiền
3. Sử dụng giọng điệu thân thiện, gần gũi
4. Sử dụng tiếng Việt
5. Không đưa ra câu hỏi của người dùng vào câu trả lời
6. Nếu không thể trả lời câu hỏi, hoặc câu hỏi không liên quan tới quản lí tài chính của người dùng, hãy trả lời "Không thể trả lời câu hỏi"
Trả lời:
    """
    
async def generate_rag(user_context: Dict, question: str) -> str:
    prompt = format_rag_prompt(question, user_context)
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            res = await client.post(
                GOOGLE_GEMINI_URL,
                json={
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                },
                params={
                    "key": GEMINI_API_KEY
                }
            )
            res.raise_for_status()
            data = res.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip().lower()
            return text
    except Exception:
        print("⚠️ Lỗi khi tạo câu trả lời:")
        traceback.print_exc()
        return "Không thể trả lời câu hỏi lúc này."