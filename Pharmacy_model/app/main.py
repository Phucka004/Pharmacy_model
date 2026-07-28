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

# --- ĐÁNH GIÁ ---
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score.scorer import BERTScorer

bert_scorer = BERTScorer(lang="vi", model_type="bert-base-multilingual-cased")

# ====================== CẤU HÌNH ======================
DB_PATH = "data/vector_db"
COLLECTION_NAME = "pharmacy_knowledge"
HISTORY_FILE = "data/chat_history_v4.json"

os.makedirs("data", exist_ok=True)

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)

llm = OllamaLLM(model="llama3.2:3b", temperature=0.0)
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

def calculate_all_metrics(reference, candidate):
    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothie)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = scorer.score(reference, candidate)['rougeL'].fmeasure
    P, R, F1 = bert_scorer.score([candidate], [reference])
    return {"bleu": round(bleu, 4), "rouge_l": round(rouge_l, 4), "bert_score": round(F1.mean().item(), 4)}

def expand_query(query):
    try:
        prompt = f"""Liệt kê 2-3 từ khóa liên quan đến: "{query}". Chỉ trả về từ khóa cách nhau bởi dấu phẩy."""
        return f"{query}, {llm.invoke(prompt).strip()}"
    except:
        return query

def refine_context(scored_candidates, limit=5):
    final_contexts = []
    seen = set()
    priority_map = {'medical': 1, 'dinhnghia': 2, 'product': 3}

    sorted_candidates = sorted(scored_candidates, key=lambda x: (priority_map.get(x[1].get('type'), 99), -x[2]))

    for doc, meta, score in sorted_candidates:
        content = meta.get('original_content', doc)
        if content in seen: continue
        seen.add(content)
        tag = meta.get('type', 'INFO').upper()
        category = meta.get('category', 'Unknown').split(':')[-1]
        final_contexts.append(f"### [{tag} - {category}]: {content}")
        if len(final_contexts) >= limit: break
    return "\n\n".join(final_contexts)


# ====================== CORE - V6.0 ======================
def ask_pharmabee(user_input, chat_history_str=""):
    start_time = time.time()

    print(f"🔍 Đang xử lý: {user_input}")

    query_no_accent = unidecode(user_input)
    expanded = expand_query(user_input)
    results = collection.query(query_texts=[user_input, expanded], n_results=15)

    all_docs, all_metas, seen_ids = [], [], set()
    for docs, metas, ids in zip(results['documents'], results['metadatas'], results['ids']):
        for doc, meta, idx in zip(docs, metas, ids):
            if idx not in seen_ids:
                all_docs.append(doc)
                all_metas.append(meta)
                seen_ids.add(idx)

    if not all_docs:
        return "Xin lỗi, PharmaBee chưa có dữ liệu về trường hợp này.", {"bleu":0, "rouge_l":0, "bert_score":0}

    pairs = [[user_input, doc] for doc in all_docs[:12]]
    scores = reranker.predict(pairs)
    scored_candidates = sorted(zip(all_docs, all_metas, scores), key=lambda x: x[2], reverse=True)

    context_str = refine_context(scored_candidates, limit=5)

    # ==================== PROMPT V6.0 - LOGIC CHẨN ĐOÁN THÔNG MINH ====================
    instruction = """
BẠN LÀ DƯỢC SĨ NHÀ THUỐC PHARMACY.

QUY TẮC CỨNG:
- Phân tích triệu chứng từ câu hỏi của khách.
- Dùng CONTEXT để tìm bệnh phù hợp nhất với các triệu chứng.
- Nếu chỉ có 1-2 triệu chứng hoặc nhiều bệnh có thể → Hỏi thêm để xác định rõ bệnh.
- Chỉ khi xác định được bệnh rõ ràng (cảm cúm, viêm họng,...) mới kê đơn combo thuốc từ CONTEXT.
- Không bịa thuốc. Không kê đơn khi chưa chắc chắn.
 Nếu khách có **sốt + đau họng + ho + sổ mũi** (hoặc gần giống) → ĐỦ ĐIỀU KIỆN. DỪNG HỎI NGAY, chẩn đoán tạm thời là **cảm cúm / sốt siêu vi**, và KÊ ĐƠN COMBO THUỐC OTC.
- TUYỆT ĐỐI KHÔNG hỏi thêm khi đã đủ 3 triệu chứng, không lặp câu hỏi, không nhắc vắc-xin.
- Kê đơn rõ ràng, có liều sáng-trưa-tối, liệu trình 3-5 ngày.
- Luôn kết thúc bằng: "Bạn nên đến gặp bác sĩ để có kết luận chính xác nhất."

QUY TẮC BẮT BUỘC (CỨNG - PHẢI TUÂN THỦ 100%):
- CHỈ sử dụng thông tin bệnh lý và tên thuốc có trong CONTEXT.
- Tuyệt đối KHÔNG bịa ra tên thuốc (Acetaminophen, Ibuprofen, Tylenol, Advil...) nếu không có trong CONTEXT.
- Nếu không tìm thấy thuốc phù hợp trong CONTEXT → Không kê đơn, chỉ khuyên đi khám bác sĩ.
- Khi kê đơn phải ghi rõ: Tên thuốc (theo CONTEXT) - Liều dùng - Liệu trình.
- Phân biệt rõ OTC và ETC.


"""

    prompt = f"""Bạn là Dược sĩ Nhà thuốc Pharmacy.

NHIỆM VỤ:
{instruction}

CONTEXT (chứa bệnh và thuốc):
{context_str}

LỊCH SỬ:
{chat_history_str if chat_history_str else "Lần đầu"}

KHÁCH HỎI: {user_input}

TRẢ LỜI:"""

    response = llm.invoke(prompt)

    # ==================== CARD SẢN PHẨM - CHỈ HIỂN THỊ KHI KÊ ĐƠN ====================
    info_extra = ""
    # Chỉ show card khi AI đã kê đơn (có từ "thuốc", "kê", "sáng", "trưa", "tối")
    if any(word in response.lower() for word in ["thuốc", "kê", "sáng", "trưa", "tối", "liều"]):
        best_product = next((m for _, m, _ in scored_candidates if m.get('type') == 'product'), None)
        if best_product:
            info_extra = f"\n\n--- 💊 SẢN PHẨM GỢI Ý ---"
            info_extra += f"\n📦 {best_product.get('product_name','N/A')}"
            info_extra += f"\n🌟 Nhóm: {best_product.get('category','N/A').split(':')[-1].replace('-',' ').title()}"
            info_extra += f"\n💰 Giá: {best_product.get('price','Liên hệ')}"
            info_extra += f"\n📋 ETC: {best_product.get('is_etc','Không')}"

    duration = time.time() - start_time
    metrics = calculate_all_metrics(context_str, response)
    save_chat_history(user_input, response, metrics, best_product.get('category') if 'best_product' in locals() else None, duration)

    return response + info_extra, metrics


# ====================== CHẠY APP ======================
if __name__ == "__main__":
    print("="*100)
    print("🐝 PHARMABEE AI V6.0 - PHÂN TÍCH TRIỆU CHỨNG THÔNG MINH")
    print("="*100)

    chat_history = []

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
            answer, _ = ask_pharmabee(u_input, history_str)
            latency = time.time() - start

            print(f"\n🤖 PharmaBee:\n{answer}")
            print(f"⏱️  Thời gian: {latency:.2f}s\n")

            main_answer = answer.split("--- 💊")[0].strip()
            chat_history.append(f"PharmaBee: {main_answer}")

            if len(chat_history) > 12:
                chat_history = chat_history[-8:]
        except Exception as e:
            print(f"❌ Lỗi: {e}")
