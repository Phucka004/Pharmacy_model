import os
import json
import chromadb
from datetime import datetime
from langchain_ollama import OllamaLLM
from sentence_transformers import CrossEncoder
from chromadb.utils import embedding_functions
from unidecode import unidecode
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score.scorer import BERTScorer

# --- KHỞI TẠO HỆ THỐNG ---
bert_scorer = BERTScorer(lang="vi", model_type="bert-base-multilingual-cased")
DB_PATH = "data/vector_db"
COLLECTION_NAME = "pharmacy_knowledge"
os.makedirs("data", exist_ok=True)

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)
llm = OllamaLLM(model="llama3.2:3b", temperature=0.0)
reranker = CrossEncoder('mixedbread-ai/mxbai-rerank-xsmall-v1')

# --- HÀM TRỢ GIÚP ---
def calculate_all_metrics(reference, candidate):
    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothie)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = scorer.score(reference, candidate)['rougeL'].fmeasure
    P, R, F1 = bert_scorer.score([candidate], [reference])
    return {"bleu": round(bleu, 4), "rouge_l": round(rouge_l, 4), "bert_score": round(F1.mean().item(), 4)}

def refine_context(scored_candidates, limit=5):
    final_contexts = []
    seen_content = set()
    priority_map = {'medical': 1, 'dinhnghia': 2, 'product': 3, 'info': 4}
    
    sorted_candidates = sorted(
        scored_candidates, 
        key=lambda x: (priority_map.get(x[1].get('type', 'info'), 99), -x[2])
    )

    for doc, meta, score in sorted_candidates:
        original = meta.get('original_content', doc)
        if original not in seen_content:
            tag = meta.get('type', 'MEDICAL').upper()
            final_contexts.append(f"### [{tag}]: {original}")
            seen_content.add(original)
        if len(final_contexts) >= limit: break
    return "\n\n".join(final_contexts)

# --- CORE LOGIC ---
def ask_pharmabee(user_input, session_symptoms):
    start_time = datetime.now()
    
    # Gom tất cả triệu chứng từ đầu buổi chat đến giờ
    combined_query = f"{session_symptoms} {user_input}"
    
    # RAG Retrieval
    results = collection.query(query_texts=[combined_query, unidecode(combined_query)], n_results=15)
    all_docs, all_metas, seen_ids = [], [], set()
    for i in range(len(results['documents'])):
        for doc, meta, idx in zip(results['documents'][i], results['metadatas'][i], results['ids'][i]):
            if idx not in seen_ids:
                all_docs.append(doc); all_metas.append(meta); seen_ids.add(idx)

    if not all_docs:
        return "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp.", {"bleu":0, "rouge_l":0, "bert_score":0}

    # Re-ranking
    pairs = [[combined_query, doc] for doc in all_docs]
    scores = reranker.predict(pairs)
    scored_candidates = sorted(zip(all_docs, all_metas, scores), key=lambda x: x[2], reverse=True)
    
    context_str = refine_context(scored_candidates)
    top_meta = scored_candidates[0][1]

    instruction = """
    BẠN LÀ DƯỢC SĨ NHÀ THUỐC PHARMACY.
    
   1. CHIẾN THUẬT THĂM KHÁM (BẮT BUỘC):
    - Khi khách nêu triệu chứng chung (ví dụ: 'Sốt'), TUYỆT ĐỐI không kết luận ngay và không nhắc đến các yếu tố bên ngoài (như vắc-xin).
    - Phải đặt câu hỏi phân loại để xác định 1 trong 3 nhóm bệnh: Sốt rét (rét run, theo cơn), Sốt siêu vi (ho, sổ mũi, đau họng), hoặc Sốt xuất huyết (phát ban, đau họng, chảy máu).
    - Chỉ khi khách xác nhận thêm triệu chứng mới được đưa ra chẩn đoán tạm thời.

    2. GIỚI HẠN KIẾN THỨC (CỨNG):
    - CHỈ sử dụng thông tin bệnh lý và thuốc có trong CONTEXT. 
    - Nếu triệu chứng lạ không có trong dữ liệu, hãy yêu cầu khách đi khám bác sĩ ngay.

    3. QUY TẮC KÊ ĐƠN & LINK ONLINE:
    - Chỉ kê đơn thuốc phối hợp cho nhóm OTC (is_etc: Không). 
    - Trình bày đơn thuốc rõ ràng: [Tên thuốc] - [Liều dùng: Sáng/Trưa/Tối] - [Liệu trình: 3-5 ngày].
    - Với thuốc kê đơn (ETC): Chỉ nêu tên và ghi 'Cần có chỉ định từ bác sĩ', tuyệt đối không đưa liều dùng.

    4. PHONG CÁCH PHẢN HỒI:
    - Gãy gọn, chuyên nghiệp, không từ chối trả lời kiểu máy móc (Không nói 'Tôi không thể kê đơn').
    - Luôn kết thúc bằng: 'Bạn nên đến gặp bác sĩ để có kết luận chính xác nhất'.
    """

    prompt = f"""{instruction}
    CONTEXT: {context_str}
    LỊCH SỬ TRIỆU CHỨNG: {session_symptoms}
    CÂU HỎI MỚI: {user_input}
    TRẢ LỜI:"""

    response = llm.invoke(prompt)

    # --- XỬ LÝ SẢN PHẨM & LINK ĐẶT HÀNG ---
    info_extra = ""
    image_url = "N/A"
    best_product = next((m for d, m, s in scored_candidates if m.get('type') == 'product'), None)

    if best_product and "đơn thuốc" in response.lower(): # Chỉ hiện card khi AI bắt đầu kê đơn
        product_name = best_product.get('product_name', 'N/A')
        is_etc = best_product.get('is_etc', 'Không')
        image_url = best_product.get('image', 'N/A')
        
        # Tạo Link đặt hàng giả định (slug hóa tên sản phẩm)
        product_slug = unidecode(product_name).lower().replace(" ", "-")
        order_link = f"https://nhathuoclongchau.com.vn/thuc-pham-chuc-nang/{product_slug}"

        info_extra = f"\n\n--- 💊 CHI TIẾT SẢN PHẨM GỢI Ý ---"
        info_extra += f"\n📦 Tên thuốc: {product_name}"
        info_extra += f"\n📋 Loại: {'Thuốc kê đơn (ETC)' if is_etc == 'Có' else 'Thuốc không kê đơn (OTC)'}"
        info_extra += f"\n🔗 Đặt mua online: {order_link}"
        if image_url != "N/A":
            info_extra += f"\n🖼️ Ảnh: {image_url}"

    metrics = calculate_all_metrics(context_str, response)
    return response + info_extra, metrics

# --- CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    print("🐝 PHARMABEE AI V5.0 - HỆ THỐNG TƯ VẤN CHUYÊN NGHIỆP")
    
    # Biến lưu trữ triệu chứng xuyên suốt session
    session_symptoms = "" 
    
    while True:
        u_input = input("\n👤 Khách hàng: ")
        if u_input.lower() in ['exit', 'quit']: break
        
        try:
            answer, m = ask_pharmabee(u_input, session_symptoms)
            print(f"\n🤖 PharmaBee: {answer}")
            
            # Cập nhật biến lưu trữ triệu chứng (Trích xuất ý chính để tránh đầy context)
            session_symptoms += f" {u_input}" 
            
        except Exception as e:
            print(f"❌ Lỗi hệ thống: {e}")