import os
import json
import time
import chromadb
import pandas as pd
from datetime import datetime
from langchain_ollama import OllamaLLM
from sentence_transformers import CrossEncoder
from chromadb.utils import embedding_functions
from unidecode import unidecode

# ====================== CẤU HÌNH ======================
DB_PATH = "data/vector_db"
COLLECTION_NAME = "pharmacy_knowledge"
HISTORY_FILE = "data/chat_history_advanced_v7.json"

os.makedirs("data", exist_ok=True)

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)

# Sử dụng mô hình Llama 3.2 3B
llm = OllamaLLM(model="llama3.2:3b", temperature=0.1) 
reranker = CrossEncoder('mixedbread-ai/mxbai-rerank-xsmall-v1')


# ====================== HÀM HỖ TRỢ ======================
def save_chat_history(query, response, metrics, best_match, duration):
    history_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_query": query,
        "ai_response": response[:700],
        "best_match": best_match,
        "metrics": metrics,
        "latency_seconds": round(duration, 2)
    }
    data = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            try: data = json.load(f)
            except: data = []
    data.append(history_entry)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def refine_context(scored_candidates, limit=4):
    """Giảm giới hạn xuống 4 tài liệu để giảm số lượng token truyền vào prompt, giúp LLM sinh chữ nhanh hơn"""
    final_contexts = []
    seen = set()
    priority_map = {'product': 1, 'medical': 2, 'dinhnghia': 3}

    valid_candidates = [x for x in scored_candidates if x[2] > 0.15]
    sorted_candidates = sorted(valid_candidates, key=lambda x: (priority_map.get(x[1].get('type'), 99), -x[2]))

    for doc, meta, score in sorted_candidates:
        content = meta.get('original_content', doc)
        if content in seen: continue
        seen.add(content)
        tag = meta.get('type', 'INFO').upper()
        category = meta.get('category', 'Unknown').split(':')[-1]
        final_contexts.append(f"### [{tag} - {category}]: {content}")
        if len(final_contexts) >= limit: break
    return "\n\n".join(final_contexts)


# ====================== CORE - V6.8 (TỐI ƯU HÓA TỐC ĐỘ TOÀN DIỆN) ======================
def ask_pharmabee(user_input, collection, reranker, chat_history_str="", last_products_str=""):
    info_extra = ""
    
    # Lấy dữ liệu và LỌC TRÙNG dựa trên tên sản phẩm
    results = collection.query(query_texts=[user_input], n_results=4)
    all_docs, all_metas, seen_keys = [], [], set()
    
    for docs, metas in zip(results['documents'], results['metadatas']):
        for doc, meta in zip(docs, metas):
            # Tạo khóa duy nhất để lọc trùng (ưu tiên tên sản phẩm)
            key = meta.get("product_name") or meta.get("name") or doc[:30]
            if key not in seen_keys:
                all_docs.append(doc)
                all_metas.append(meta)
                seen_keys.add(key)

    # Reranking
    pairs = [[user_input, doc] for doc in all_docs]
    scores = reranker.predict(pairs) if pairs else []
    scored_candidates = sorted(zip(all_docs, all_metas, scores), key=lambda x: x[2], reverse=True)

    # Nếu lượt trước đã có gợi ý sản phẩm, giữ nguyên
    if last_products_str:
        info_extra = last_products_str
    else:
        # Lượt đầu: Tạo danh sách gợi ý sản phẩm
        products = [x for x in scored_candidates if x[1].get('type') == 'product' and x[2] > 0.15]
        if products:
            info_extra = "\n\n--- 💊 CÁC SẢN PHẨM GỢI Ý ---\n"
            for doc, meta, score in products[:3]:
                info_extra += f"📦 {meta.get('product_name','N/A')} - Giá: {meta.get('price','Liên hệ')}\n"

        context_str = refine_context(scored_candidates, limit=4)
        info_extra = "\n\n--- 💊 CÁC SẢN PHẨM GỢI Ý ---"
        product_count = 0
        for doc, meta, score in scored_candidates:
            if meta.get('type') == 'product' and score > 0.15 and product_count < 4:
                info_extra += f"\n\n📦 {meta.get('product_name','N/A')}"
                info_extra += f"\n🌟 Nhóm: {meta.get('category','N/A').split(':')[-1].replace('-',' ').title()}"
                info_extra += f"\n💰 Giá: {meta.get('price','Liên hệ')}"
                info_extra += f"\n📋 ETC: {meta.get('is_etc','Không')}"
                product_count += 1
                
        if product_count == 0:
            info_extra = ""

    # ==================== PROMPT V6.8 (TỔNG QUÁT & GỌN GÀNG) ====================
    instruction = """
BẠN LÀ MỘT DƯỢC SĨ TƯ VẤN CHUYÊN NGHIỆP TẠI NHÀ THUỐC.
Hãy giao tiếp ân cần, đáng tin cậy, ngắn gọn và tập trung giải quyết vấn đề sức khỏe của khách hàng.

QUY TẮC AN TOÀN TUYỆT ĐỐI & ĐIỀU HƯỚNG TỪ CHỐI (QUAN TRỌNG):
- CHỈ TRẢ LỜI dựa trên các triệu chứng hoặc thông tin sức khỏe khách hàng CHỦ ĐỘNG cung cấp.
- TUYỆT ĐỐI KHÔNG tự suy diễn, gán ghép thêm các bệnh lý nghiêm trọng nằm ngoài câu hỏi.
- Nếu thông tin quá chung chung, hãy tư vấn khái quát và đặt câu hỏi ngắn gọn để làm rõ tình trạng.
- KHÔNG ĐƯỢC từ chối bằng các câu máy móc như: "Tôi không thể kê đơn", "Tôi không có thẩm quyền bốc thuốc". Khi khách hàng yêu cầu "kê đơn", "cho liều uống x ngày" đối với các bệnh thông thường (cảm cúm, sốt, ho, sổ mũi), hãy hiểu đó là yêu cầu: "Đề xuất lộ trình phối hợp sử dụng các sản phẩm OTC hiện có trong CONTEXT".

QUY TẮC PHÁT TRIỂN HỘI THOẠI VÀ CHỐNG LẶP Ý:
- Kiểm tra kỹ LỊCH SỬ chat để nắm tiến trình tư vấn.
- Nếu ở lượt trước bạn đã đặt câu hỏi khai thác thông tin và ở lượt này khách hàng ĐÃ cung cấp câu trả lời -> DỪNG HỎI LẠI NGAY LẬP TỨC.
- Chuyển hướng trực tiếp sang việc phân tích dữ kiện mới nhận, đưa ra giải pháp chăm sóc y tế hoặc liệt kê các sản phẩm phù hợp có trong CONTEXT.
- Không nhắc đi nhắc lại một câu cảnh báo hoặc một cụm từ khuôn mẫu liên tục trong cùng một lượt nói khiến văn phong bị máy móc.

QUY TẮC TƯ VẤN SẢN PHẨM & THUỐC (ÉP FORM ĐƠN THUỐC):
- Phân tích triệu chứng từ câu hỏi của khách. Dùng CONTEXT để tìm bệnh phù hợp nhất. Không bịa thuốc.
- Nếu khách có đủ combo triệu chứng (ví dụ: sốt + đau họng + ho + sổ mũi hoặc gần giống) HOẶC khách chủ động yêu cầu cho đơn/lộ trình uống x ngày -> Lập tức đưa ra chẩn đoán tạm thời (ví dụ: Cảm cúm / Viêm đường hô hấp trên) và ĐỀ XUẤT COMBO SẢN PHẨM OTC rõ ràng từ CONTEXT.
- Định dạng đơn thuốc bắt buộc ghi rõ theo dạng gạch đầu dòng: Tên sản phẩm/Thuốc (Theo CONTEXT) - Liều dùng (Sáng/Trưa/Tối) - Liệu trình số ngày đúng theo yêu cầu của khách (Ví dụ: 3 ngày).
- Phân biệt rõ OTC (Đề xuất liều lượng cụ thể) và ETC (Chỉ nêu tên tham khảo từ CONTEXT nếu có liên quan và nhấn mạnh phải theo chỉ định trực tiếp của bác sĩ).
- Chỉ tư vấn sản phẩm ĐÚNG mục đích điều trị triệu chứng của khách. Không đưa các sản phẩm lệch nhóm vào đơn thuốc.
- TUYỆT ĐỐI KHÔNG hỏi thêm khi đã đủ dữ kiện triệu chứng hoặc khi khách đã chốt yêu cầu lấy thuốc.
- Luôn kết thúc câu trả lời bằng ĐÚNG một câu chốt duy nhất ở cuối cùng bài nói: "Bạn nên đến gặp bác sĩ để có kết luận chính xác nhất."
"""

    prompt = f"""Bạn là Dược sĩ Nhà thuốc Pharmacy.

NHIỆM VỤ:
{instruction}

CONTEXT:
{context_str}

LỊCH SỬ:
{chat_history_str if chat_history_str else "Lần đầu"}

KHÁCH HỎI: {user_input}

TRẢ LỜI NGẮN GỌN (ĐI THẲNG VÀO COMBO THUỐC OTC NẾU ĐỦ ĐIỀU KIỆN):"""

    return prompt, info_extra


# ====================== CHẠY APP ======================
if __name__ == "__main__":
    print("="*100)
    print("🐝 PHARMABEE AI V7.0 - TỐI ƯU TỐC ĐỘ ĐA NĂNG")
    print("="*100)

    chat_history = []
    last_saved_products = "" 

    while True:
        u_input = input("\n👤 Khách hàng: ").strip()
        if u_input.lower() in ['exit', 'quit', 'thoát']:
            print("Cảm ơn bạn đã sử dụng PharmaBee!")
            break
        if not u_input:
            continue

        chat_history.append(f"Khách: {u_input}")
        history_str = "\n".join(chat_history[-6:])

        try:
            start = time.time()
            print("🤖 PharmaBee đang suy nghĩ...")
            
            # 1. Gọi hàm để lấy chuỗi Prompt thô và thông tin extra chân trang đã đóng băng
            prompt_str, current_products = ask_pharmabee(u_input, history_str, last_saved_products)
            
            if current_products and not last_saved_products:
                last_saved_products = current_products

            print("\n🤖 PharmaBee: ", end="", flush=True)
            
            # 2. KÍCH HOẠT CHẾ ĐỘ STREAMING: Chữ ra đến đâu, in ra màn hình ngay lập tức đến đó
            full_response = ""
            for chunk in llm.stream(prompt_str):
                print(chunk, end="", flush=True)
                full_response += chunk
                
            # 3. In danh sách sản phẩm gợi ý ra sau cùng khi chữ đã chạy xong
            if last_saved_products:
                print(last_saved_products)
            elif current_products:
                print(current_products)

            latency = time.time() - start
            print(f"\n⏱️  Thời gian phản hồi toàn cục: {latency:.2f}s\n")

            # 4. Lưu lịch sử dựa trên nội dung text thực tế đã stream ra
            chat_history.append(f"PharmaBee: {full_response.strip()}")
            save_chat_history(u_input, full_response, None, None, latency)

            if len(chat_history) > 12:
                chat_history = chat_history[-8:]

        except Exception as e:
            print(f"❌ Lỗi: {e}")