LLAMA_HOST = "https://delicate-prawn-randomly.ngrok-free.app/api"
# LLAMA_HOST = "http://192.168.1.87:11434/api"

LLAMA_GENERATE_API_URL = f"{LLAMA_HOST}/generate"  # Địa chỉ API của Ollama cho Llama 3.2
LLAMA_CHAT_API_URL = f"{LLAMA_HOST}/chat"  # Địa chỉ API của Ollama cho Llama 3.2
MODEL_NAME = "llama3"
CACHE_MAX_SIZE = 100  # Kích thước cache cho các hàm gọi Llama 3.2
DB_PATH = "finance_v2.db"  # Đường dẫn cơ sở dữ liệu SQLite
GEMINI_API_KEY = "AIzaSyCCsuoRfyhdeMLKyuzi4ae-aUsCKT5ivoQ"

DEFAULT_CURRENCY_ID="859dbc6a-63ee-4be2-983b-e1f5528ce5e4"

GOOGLE_GEMINI_URL="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
