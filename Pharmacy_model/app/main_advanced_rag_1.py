import os
import json
import time
import torch
import chromadb
from datetime import datetime
from langchain_ollama import OllamaLLM
from sentence_transformers import CrossEncoder
from chromadb.utils import embedding_functions

# ====================== CẤU HÌNH & TỐI ƯU ======================
DB_PATH = "data/vector_db"
COLLECTION_NAME = "pharmacy_knowledge"

# Tự động chọn GPU nếu có
device = 'cuda' if torch.cuda.is_available() else 'cpu'
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME, 
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2"))

llm = OllamaLLM(model="llama3.2:3b", temperature=0.1) 
reranker = CrossEncoder('mixedbread-ai/mxbai-rerank-xsmall-v1', device=device)

# Cache biến toàn cục
cached_scored_candidates = []

# BỘ NÃO PHÂN LOẠI Ý ĐỊNH
def get_intent(user_input):
    prompt = f"Phân loại câu: '{user_input}'. Trả về 1 từ: SYMPTOM, PURCHASE, hoặc OTHER. Không giải thích."
    return llm.invoke(prompt).strip().upper()

def ask_pharmabee(user_input, collection, reranker, chat_history_str="", last_products_str="", turn_count=0):
    global cached_scored_candidates
    info_extra = last_products_str
    
    # TRUY XUẤT (Chỉ Rerank lượt 1)
    if turn_count == 1:
        results = collection.query(query_texts=[user_input], n_results=5)
        all_docs, all_metas, seen_keys = [], [], set()
        for docs, metas in zip(results['documents'], results['metadatas']):
            for doc, meta in zip(docs, metas):
                key = meta.get("product_name") or meta.get("name") or doc[:20]
                if key not in seen_keys:
                    all_docs.append(doc); all_metas.append(meta); seen_keys.add(key)
        
        pairs = [[user_input, doc] for doc in all_docs[:4]]
        scores = reranker.predict(pairs) if pairs else []
        cached_scored_candidates = sorted(zip(all_docs, all_metas, scores), key=lambda x: x[2], reverse=True)
    
    scored_candidates = cached_scored_candidates

    # GỢI Ý SẢN PHẨM
    if turn_count == 1:
        products = [x for x in scored_candidates if x[1].get('type') == 'product' and x[2] > 0.15]
        if products:
            info_extra = "\n\n--- 💊 GỢI Ý ---"
            for _, meta, _ in products[:2]:
                info_extra += f"\n📦 {meta.get('product_name')} - {meta.get('price')}"

    # PROMPT RÚT GỌN (Tăng tốc độ)
    context_str = "\n\n".join([x[0] for x in scored_candidates[:2]])
    instruction = "Bạn là Dược sĩ. Tư vấn dựa trên CONTEXT. KHÔNG nhắc thuốc ngoài CONTEXT. Ngắn (<80 chữ). Chốt: 'Bạn nên đến gặp bác sĩ để có kết luận chính xác nhất.'"
    
    prompt = f"{instruction}\n\nCONTEXT: {context_str}\n\nLỊCH SỬ: {chat_history_str}\n\nKHÁCH: {user_input}\n\nTRẢ LỜI:"
    
    return prompt, info_extra

# ====================== VÒNG LẶP CHÍNH ======================
if __name__ == "__main__":
    chat_history = []
    last_saved_products = "" 
    turn_counter = 0

    while True:
        u_input = input("\n👤 Khách: ").strip()
        if u_input.lower() in ['exit', 'thoát']: break
        if not u_input: continue

        # 1. LỌC Ý ĐỊNH BẰNG LLM
        intent = get_intent(u_input)
        if intent == "OTHER":
            print("🤖 PharmaBee: Tôi xin lỗi, tôi chỉ hỗ trợ thông tin thuốc và sức khỏe.")
            continue

        # 2. XỬ LÝ RAG
        turn_counter += 1
        start = time.time()
        
        prompt_str, current_products = ask_pharmabee(u_input, collection, reranker, 
                                                    "\n".join(chat_history[-2:]), 
                                                    last_saved_products, turn_counter)
        last_saved_products = current_products
        
        print("🤖 PharmaBee: ", end="", flush=True)
        full_response = ""
        for chunk in llm.stream(prompt_str):
            print(chunk, end="", flush=True)
            full_response += chunk
        
        print(last_saved_products)
        print(f"\n⏱️ Tốc độ: {time.time()-start:.2f}s")

        chat_history.append(f"Khách: {u_input}\nPharmaBee: {full_response}")
        
        # 3. RESET SAU 2 LƯỢT
        if turn_counter >= 2:
            print("\n--- [RESET: Đã làm sạch gợi ý cho phiên mới] ---")
            turn_counter = 0
            last_saved_products = ""
            chat_history = []
            cached_scored_candidates = []