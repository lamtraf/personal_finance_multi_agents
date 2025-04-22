LLAMA_HOST = "https://delicate-prawn-randomly.ngrok-free.app/api"

LLAMA_GENERATE_API_URL = f"{LLAMA_HOST}/generate"  # Địa chỉ API của Ollama cho Llama 3.2
LLAMA_CHAT_API_URL = f"{LLAMA_HOST}/chat"  # Địa chỉ API của Ollama cho Llama 3.2
MODEL_NAME = "llama3.2"
CACHE_MAX_SIZE = 100  # Kích thước cache cho các hàm gọi Llama 3.2
DB_PATH = "finance_v2.db"  # Đường dẫn cơ sở dữ liệu SQLite