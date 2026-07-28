# ⚙️ CẤU HÌNH CHÍNH CHO HỆ THỐNG DƯỢC SĨ AI

import os
from pathlib import Path

# 📂 ĐƯỜNG DẪN
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
LOGS_DIR = BASE_DIR / "logs"
HISTORY_FILE = BASE_DIR / "conversation_history.json"

# 🤖 CẤU HÌNH MÔ HÌNH NGÔN NGỮ & EMBEDDING
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Mô hình nhúng từ HuggingFace
LLM_MODEL = "llama3"  # Mô hình LLM từ Ollama
LLM_TEMPERATURE = 0.1  # 0-1: càng thấp->càng hay lặp lại; càng cao->càng sáng tạo

# 📊 CẤU HÌNH RETRIEVAL
RETRIEVER_K = 5  # Số tài liệu top được lấy ra
CHUNK_SIZE = 1000  # Kích thước chunk (token)
CHUNK_OVERLAP = 200  # Độ chồng lấp giữa các chunk

# 💾 CẤU HÌNH CHROMADB
CHROMA_COLLECTION_NAME = "pharmacy_products"
CHROMA_METRIC = "cosine"  # "cosine", "l2", "ip"

# 🔧 CẤU HÌNH HỆ THỐNG
DEBUG = False
LOG_LEVEL = "INFO"
MAX_HISTORY_RECORDS = 1000  # Số lịch sử tối đa lưu trữ

# 🌍 CẤU HÌNH NGÔN NGỮ
LANGUAGE = "vi"  # Vietnamese
SUPPORTED_LANGUAGES = ["vi", "en"]

# ✅ Tạo thư mục nếu chưa có
for directory in [DATA_DIR, CHROMA_DB_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)
