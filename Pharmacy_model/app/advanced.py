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
reranker = CrossEncoder('mixedbread-ai/mxbai-rerank-xsmall-v1', device='cpu')


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

    # CHÚ Ý: Nâng ngưỡng lọc lên 0.30 để tránh lọt tài liệu rác gây loãng thông tin
    valid_candidates = [x for x in scored_candidates if x[2] > 0.30]
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

# ====================== BỘ LỌC Ý ĐỊNH BẰNG OLLAMA (LLM GUARDRAIL) ======================
def classify_intent_by_llm(user_input):
    """
    Sử dụng Ollama để phân loại ý định của khách hàng.
    Ép mô hình chỉ trả về đúng 1 từ duy nhất: MEDICAL hoặc CHITCHAT để xử lý bằng code.
    """
    intent_prompt = f"""Bạn là bộ lọc phân loại ý định câu hỏi cho hệ thống Y tế.
Nhiệm vụ của bạn là đọc câu hỏi của khách hàng và phân loại vào 1 trong 2 nhóm:
- MEDICAL: Nếu câu hỏi liên quan đến sức khỏe, triệu chứng bệnh, thuốc, cách điều trị, hoặc nhà thuốc.
- CHITCHAT: Nếu câu hỏi là tán tỉnh, chào hỏi suông, hỏi tuổi tác, thời tiết, rủ rê kết bạn, hoặc các chủ đề ngoài y tế.

Quy tắc tối cao: Chỉ trả về đúng 1 từ duy nhất là 'MEDICAL' hoặc 'CHITCHAT'. Không giải thích, không viết thêm từ nào khác.

Câu hỏi của khách hàng: "{user_input}"

Phân loại (CHỈ VIẾT 1 TỪ):"""
    
    try:
        # Gọi Ollama chạy riêng một lượt prompt siêu ngắn để check intent
        response = llm.invoke(intent_prompt).strip().upper()
        # Clean chuỗi trả về phòng trường hợp LLM sinh dư dấu chấm, khoảng trắng
        if "MEDICAL" in response:
            return "MEDICAL"
        else:
            return "CHITCHAT"
    except Exception as e:
        print(f"⚠️ Lỗi bộ lọc Intent LLM: {e}. Mặc định cho qua.")
        return "MEDICAL"

# ====================== CORE - V7.5 (TỐI ƯU HÓA KHUÔN MẪU ĐA BỆNH) ======================
# ====================== CORE - V9.0 (CẮT ĐUÔI ẢO GIÁC) ======================
def ask_pharmabee(user_input, collection, reranker, chat_history_str="", last_products_str="", turn_count=1):
    print(f"🔍 Đang xử lý (Lượt {turn_count} - Phiên bản V9.0): {user_input}")
    products_context = ""

    if last_products_str:
        print("📌 Phát hiện sản phẩm từ lượt trước, giữ nguyên danh sách gợi ý cũ để chống nhiễu.")
        info_extra = last_products_str
        products_context = last_products_str.replace("--- 💊 CÁC SẢN PHẨM GỢI Ý ---", "").strip()
        
        results = collection.query(query_texts=[user_input], n_results=4)
        all_docs, all_metas, seen_ids = [], [], set()
        for docs, metas, ids in zip(results['documents'], results['metadatas'], results['ids']):
            for doc, meta, idx in zip(docs, metas, ids):
                if idx not in seen_ids and meta.get('type') != 'product':
                    all_docs.append(doc)
                    all_metas.append(meta)
                    seen_ids.add(idx)
                    
        pairs = [[user_input, doc] for doc in all_docs[:5]]
        scores = reranker.predict(pairs) if pairs else []
        scored_candidates = sorted(zip(all_docs, all_metas, scores), key=lambda x: x[2], reverse=True) if pairs else []
        context_str = refine_context(scored_candidates, limit=3)
    else:
        results = collection.query(query_texts=[user_input], n_results=5)
        all_docs, all_metas, seen_ids = [], [], set()
        for docs, metas, ids in zip(results['documents'], results['metadatas'], results['ids']):
            for doc, meta, idx in zip(docs, metas, ids):
                if idx not in seen_ids:
                    all_docs.append(doc)
                    all_metas.append(meta)
                    seen_ids.add(idx)

        pairs = [[user_input, doc] for doc in all_docs[:5]]
        scores = reranker.predict(pairs) if pairs else []
        scored_candidates = sorted(zip(all_docs, all_metas, scores), key=lambda x: x[2], reverse=True) if pairs else []
        context_str = refine_context(scored_candidates, limit=3)

        info_extra = "\n\n--- 💊 CÁC SẢN PHẨM GỢI Ý ---"
        product_count = 0
        for doc, meta, score in scored_candidates:
            if meta.get('type') == 'product' and score > 0.12 and product_count < 2:
                p_name = meta.get('product_name', 'N/A')
                p_cate = meta.get('category', 'N/A').split(':')[-1].replace('-', ' ').title()
                p_price = meta.get('price', 'Liên hệ')
                
                info_extra += f"\n\n📦 {p_name}\n🌟 Nhóm: {p_cate}\n💰 Giá: {p_price}\n📋 ETC: {meta.get('is_etc','Không')}"
                products_context += f"- Thuốc đang có ở tiệm: {p_name} (Nhóm thuốc: {p_cate})\n"
                product_count += 1
                
        if product_count == 0:
            info_extra = ""
        instruction = """
1. PHÂN LOẠI: Câu hỏi ngoài y tế -> Từ chối khéo. Câu hỏi bệnh lý -> Tư vấn dựa trên CONTEXT.
2. THU THẬP: Hỏi thêm cho đến khi đủ 3 triệu chứng. Nếu đủ -> Dừng hỏi.
3. PHẢN HỒI: Chẩn đoán tạm thời (theo CONTEXT) -> Combo thuốc OTC -> Lời khuyên gặp bác sĩ.
4. ĐẢM BẢO: Không lặp ý, không bịa thuốc, không hỏi lại những gì đã có trong LỊCH SỬ.
"""

    # ==================== THAY ĐỔI LỚN: PROMPT TRẦN TRỤI, BÌNH DÂN HÓA BẬC CAO ====================
    if turn_count == 1:
        prompt = f"""Dưới đây là thông tin y tế hữu ích:
{context_str}

Hãy đóng vai là một người Dược sĩ đang đứng bán thuốc ở tiệm Pharmacy. Có một người khách hàng vừa bước vào tiệm và nói: "{user_input}"

Yêu cầu: Hãy trả lời người khách đó bằng giọng điệu ân cần, lịch sự. Tuyệt đối không từ chối. Hãy hỏi thêm đúng 1 câu hỏi phụ để biết rõ hơn tình trạng đau hoặc triệu chứng đi kèm của họ (Ví dụ: Bạn bị lâu chưa, có kèm triệu chứng nào khác không?). Viết ngắn gọn trong 2 câu. Không viết dài dòng.

Dược sĩ trả lời khách hàng:"""
    else:
        prompt = f"""Dưới đây là thông tin y tế hữu ích:
{context_str}

NHIỆM VỤ:
{instruction}

Lịch sử trò chuyện trước đó giữa bạn và khách:
{chat_history_str}

Bây giờ, khách hàng nói thêm: "{user_input}"

Yêu cầu bắt buộc: Bạn hãy đóng vai Dược sĩ, lập tức đưa ra đơn thuốc điều trị dựa vào danh sách thuốc có sẵn ở trên. Không được đặt thêm câu hỏi nào nữa. Hãy viết câu trả lời theo đúng mẫu dưới đây và điền tên thuốc thích hợp vào, không viết thêm bất kỳ điều gì khác ngoài mẫu:

Chào bạn, dựa trên triệu chứng tôi đề xuất sử dụng sản phẩm sau:
* [Điền tên thuốc có sẵn ở trên] - Liều dùng ngày 2 lần - Lộ trình: 3 ngày.
Bạn nên đến gặp bác sĩ để có kết luận chính xác nhất.

Dược sĩ trả lời khách hàng:"""

    return prompt, info_extra
# # ====================== CHẠY APP ======================
# if __name__ == "__main__":
#     print("="*100)
#     print("🐝 PHARMABEE AI V7.5 - PHIÊN BẢN CHUẨN HOÁ KHÓA LUẬN TỐT NGHIỆP")
#     print("="*100)

#     chat_history = []
#     last_saved_products = "" 
#     turn_count = 0

#     while True:
#         u_input = input("\n👤 Khách hàng: ").strip()
#         if u_input.lower() in ['exit', 'quit', 'thoát']:
#             print("Cảm ơn bạn đã sử dụng PharmaBee!")
#             break
#         if not u_input:
#             continue

#         turn_count += 1
#         chat_history.append(f"Khách: {u_input}")

#         # Cơ chế reset danh sách sản phẩm cũ ở lần thứ 3 trở đi
#         if turn_count >= 3:
#             print("🔄 Đã qua 2 lượt hội thoại, hệ thống tiến hành tự động giải phóng và reset danh sách sản phẩm cũ.")
#             last_saved_products = ""  
#             turn_count = 1  # Đặt lại turn_count về 1 cho chủ đề bệnh lý mới để kích hoạt State 1

#         # Tối ưu bộ nhớ trượt: Chỉ lấy tối đa 2 lượt gần nhất (tương đương 4 dòng tin nhắn: 2 Khách, 2 AI)
#         recent_history = chat_history[-4:] if len(chat_history) > 1 else chat_history
        
#         # Định dạng lại chuỗi lịch sử sạch đẹp để đưa vào prompt
#         history_str = ""
#         for i in range(0, len(recent_history) - 1, 2):
#             if i + 1 < len(recent_history):
#                 history_str += f"- Khách hỏi: {recent_history[i].replace('Khách: ', '')}\n"
#                 history_str += f"- Dược sĩ đã đáp: {recent_history[i+1].replace('PharmaBee: ', '')}\n"

#         try:
#             start = time.time()
#             print("🤖 PharmaBee đang suy nghĩ...")
            
#             prompt_str, current_products = ask_pharmabee(
#                 user_input=u_input, 
#                 collection=collection, 
#                 reranker=reranker, 
#                 chat_history_str=history_str, 
#                 last_products_str=last_saved_products,
#                 turn_count=turn_count
#             )
            
#             if current_products and not last_saved_products:
#                 last_saved_products = current_products

#             print("\n🤖 PharmaBee: ", end="", flush=True)
            
#             # KÍCH HOẠT CHẾ ĐỘ STREAMING
#             full_response = ""
#             for chunk in llm.stream(prompt_str):
#                 print(chunk, end="", flush=True)
#                 full_response += chunk
                
#             # In danh sách sản phẩm gợi ý ra sau cùng khi chữ đã chạy xong
#             if last_saved_products:
#                 print(last_saved_products)
#             elif current_products:
#                 print(current_products)

#             latency = time.time() - start
#             print(f"\n⏱️  Thời gian phản hồi toàn cục: {latency:.2f}s\n")

#             # Lưu lịch sử dựa trên nội dung text thực tế đã stream ra
#             chat_history.append(f"PharmaBee: {full_response.strip()}")
#             save_chat_history(u_input, full_response, None, None, latency)

#             if len(chat_history) > 8:
#                 chat_history = chat_history[-4:]

#         except Exception as e:
#             print(f"❌ Lỗi: {e}")
# ====================== CHẠY APP ======================
if __name__ == "__main__":
    print("="*100)
    print("🐝 PHARMABEE AI V8.5 - PHIÊN BẢN FIX TRIỆT ĐỂ LỖI CHẬP MẠCH CHITCHAT")
    print("="*100)

    chat_history = [] # Chứa các dictionary {"role": "user/assistant", "text": "..."} để quản lý lịch sử chuẩn
    last_saved_products = "" 
    turn_count = 0

    while True:
        u_input = input("\n👤 Khách hàng: ").strip()
        if u_input.lower() in ['exit', 'quit', 'thoát']:
            break
        if not u_input:
            continue

        # 1. KÍCH HOẠT BỘ LỌC Ý ĐỊNH BẰNG OLLAMA
        intent = classify_intent_by_llm(u_input)
        
        if intent == "CHITCHAT":
            print("\n🤖 PharmaBee: Dạ, em là Dược sĩ tư vấn y tế ảo của nhà thuốc Pharmacy. Em chỉ có thể hỗ trợ các vấn đề liên quan đến sức khỏe, triệu chứng bệnh và thuốc OTC. Anh/chị vui lòng cung cấp tình trạng sức khỏe để em hỗ trợ nhé!")
            continue # CHẶN HOÀN TOÀN, không tăng turn_count, không lưu vào lịch sử chat bẩn

        turn_count += 1
        chat_history.append({"role": "user", "text": u_input})

        # 2. CƠ CHẾ RESET KHI ĐỔI CA BỆNH MỚI
        if turn_count >= 3:
            print("🔄 Đã qua 2 lượt hội thoại, hệ thống tự động RESET để tiếp nhận ca bệnh mới.")
            last_saved_products = ""  
            chat_history = [{"role": "user", "text": u_input}]  # Giữ lại duy nhất câu hỏi đầu của ca mới
            turn_count = 1  

        # 3. ĐỊNH DẠNG LẠI LỊCH SỬ CHÁT TRỰC QUAN (Sửa lỗi lệch pha tin nhắn)
        history_str = ""
        recent_history = chat_history[-4:] if len(chat_history) > 1 else chat_history
        for msg in recent_history[:-1]: # Không lấy câu vừa gõ
            if msg["role"] == "user":
                history_str += f"- Khách hỏi: {msg['text']}\n"
            else:
                history_str += f"- Dược sĩ đáp: {msg['text']}\n"

        try:
            start = time.time()
            print("🤖 PharmaBee đang suy nghĩ...")
            
            prompt_str, current_products = ask_pharmabee(
                user_input=u_input, 
                collection=collection, 
                reranker=reranker, 
                chat_history_str=history_str, 
                last_products_str=last_saved_products,
                turn_count=turn_count
            )
            
            if current_products and not last_saved_products:
                last_saved_products = current_products

            print("\n🤖 PharmaBee: ", end="", flush=True)
            
            full_response = ""
            for chunk in llm.stream(prompt_str):
                print(chunk, end="", flush=True)
                full_response += chunk
                
            # Nếu là Lượt 1 (Mới reset hoặc bắt đầu ca bệnh mới)
            if turn_count == 1:
                # Ép hệ thống cập nhật và in danh sách thuốc mới tinh dựa trên câu hỏi mới của khách
                last_saved_products = current_products
                if last_saved_products:
                    print(last_saved_products)
            else:
                # Nếu là Lượt 2, giữ nguyên danh sách đã khóa ở Lượt 1 để chống nhiễu
                if last_saved_products:
                    print(last_saved_products)

            latency = time.time() - start
            print(f"\n⏱️  Thời gian phản hồi toàn cục: {latency:.2f}s\n")

            # Lưu vào lịch sử sạch dạng Dictionary cấu trúc
            chat_history.append({"role": "assistant", "text": full_response.strip()})

        except Exception as e:
            print(f"❌ Lỗi: {e}")