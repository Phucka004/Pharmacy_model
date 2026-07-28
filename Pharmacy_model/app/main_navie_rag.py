import os
import json
import time
import chromadb
import pandas as pd
from datetime import datetime
from langchain_ollama import OllamaLLM
from chromadb.utils import embedding_functions

# ====================== CẤU HÌNH ======================
DB_PATH = "data/vector_db"
COLLECTION_NAME = "pharmacy_knowledge"
HISTORY_FILE_NAIVE = "data/chat_history_naive_v6.json"

os.makedirs("data", exist_ok=True)

# Sử dụng chung một Embedding Function để đảm bảo tính công bằng khi đối sánh truy xuất
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)

# Giữ nguyên cấu hình mô hình Llama 3.2 3B
llm = OllamaLLM(model="llama3.2:3b", temperature=0.0)


# ====================== HÀM HỖ TRỢ GHI LOG NAIVE ======================
def save_naive_chat_history(query, response, duration):
    history_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_query": query,
        "ai_response": response[:700],
        "architecture": "Naive RAG",
        "latency_seconds": round(duration, 2)
    }
    data = []
    if os.path.exists(HISTORY_FILE_NAIVE):
        with open(HISTORY_FILE_NAIVE, 'r', encoding='utf-8') as f:
            try: data = json.load(f)
            except: data = []
    data.append(history_entry)
    with open(HISTORY_FILE_NAIVE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ====================== CORE - BASELINE (NAIVE RAG CƠ BẢN) ======================
def ask_pharmabee_naive(user_input, chat_history_str=""):
    start_time = time.time()

    print(f"🔍 [Naive RAG] Đang xử lý truy vấn thô: {user_input}")

    # NAIVE RAG: Truy xuất thẳng Top 5 văn bản thô từ ChromaDB dựa trên khoảng cách vector bề mặt
    # Không mở rộng từ khóa, không chạy qua mô hình Reranker CrossEncoder lọc nhiễu
    results = collection.query(query_texts=[user_input], n_results=5)
    
    context_docs = results['documents'][0]
    
    # Nối trực tiếp toàn bộ chuỗi văn bản bốc từ DB lên, không kiểm tra ngưỡng score, không lọc trùng
    context_str = "\n\n".join(context_docs)

    # PROMPT KIỂU CŨ: Ràng buộc lỏng lẻo, không cấu hình đánh lừa bộ lọc bảo mật y tế của Llama
    instruction = """
BẠN LÀ DƯỢC SĨ NHÀ THUỐC PHARMACY.
Hãy đọc kỹ ngữ cảnh (CONTEXT) để trả lời câu hỏi của khách hàng.
Luôn kết thúc bằng: "Bạn nên đến gặp bác sĩ để có kết luận chính xác nhất."
"""

    prompt = f"""Bạn là Dược sĩ Nhà thuốc Pharmacy.

NHIỆM VỤ:
{instruction}

CONTEXT:
{context_str}

LỊCH SỬ:
{chat_history_str if chat_history_str else "Lần đầu"}

KHÁCH HỎI: {user_input}

TRẢ LỜI:"""

    # Gọi xử lý bằng invoke() truyền thống, block hệ thống đợi sinh toàn bộ khối văn bản
    response = llm.invoke(prompt)
    duration = time.time() - start_time
    
    # Ghi log dữ liệu thực nghiệm
    save_naive_chat_history(user_input, response, duration)

    return response, duration


# ====================== CHẠY APP NAIVE DISỐI CHỨNG ======================
if __name__ == "__main__":
    print("="*100)
    print("⚠️  PHARMABEE BASELINE - KIẾN TRÚC NAIVE RAG CƠ BẢN ⚠️")
    print("="*100)

    chat_history = []

    while True:
        u_input = input("\n👤 Khách hàng: ").strip()
        if u_input.lower() in ['exit', 'quit', 'thoát']:
            print("Đã đóng hệ thống Naive RAG.")
            break
        if not u_input:
            continue

        chat_history.append(f"Khách: {u_input}")
        history_str = "\n".join(chat_history[-6:])

        try:
            print("🤖 PharmaBee (Naive) đang xử lý luồng cơ bản...")
            
            # Thực thi tác vụ truy xuất và sinh chữ tuyến tính
            answer, latency = ask_pharmabee_naive(u_input, history_str)

            print(f"\n🤖 PharmaBee (Naive):\n{answer}")
            print(f"⏱️  Thời gian phản hồi hệ thống cũ: {latency:.2f}s\n")

            chat_history.append(f"PharmaBee: {answer.strip()}")

            if len(chat_history) > 12:
                chat_history = chat_history[-8:]

        except Exception as e:
            print(f"❌ Lỗi: {e}")