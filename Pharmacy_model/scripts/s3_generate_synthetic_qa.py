import os
import json
import hashlib
import time
import argparse
import sys
import io

# --- ĐẢM BẢO ENCODING TIẾNG VIỆT TRÊN WINDOWS CONSOLE ---
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', write_through=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', write_through=True)

import pandas as pd
from tqdm import tqdm
from unidecode import unidecode
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions

# --- 1. CẤU HÌNH API GEMINI ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Sử dụng mô hình gemini-2.5-flash vì gemini-2.0-flash không có quota (limit 0) với key miễn phí này
MODEL_NAME = "gemini-2.5-flash"
model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    generation_config={"response_mime_type": "application/json", "temperature": 0.1}
)

# --- 2. CÁC ĐƯỜNG DẪN DỮ LIỆU ---
MEDICAL_CSV = "data/silver/medical_kb.csv"
PRODUCTS_CSV = "data/silver/products_kb.csv"
DB_PATH = "data/vector_db"
MEDICAL_QA_JSON = "data/silver/synthetic_medical_qa.json"
PRODUCTS_QA_JSON = "data/silver/synthetic_products_qa.json"

# --- 3. PROMPT CHUẨN ĐÃ ĐƯỢC TỐI ƯU CHO ĐỊNH DẠNG JSON OBJECT ---
MEDICAL_PROMPT = """Bạn là một chuyên gia giả lập dữ liệu y tế. Dựa trên thông tin bệnh lý được cung cấp, hãy đóng vai các bệnh nhân khác nhau (người già, người trẻ, người dùng từ địa phương) để sinh ra đúng 5 câu hỏi thực tế, mang tính khẩu ngữ đời sống mà họ sẽ hỏi dược sĩ.

QUY TẮC CỐ ĐỊNH:
1. Sử dụng văn phong giao tiếp, ngôn ngữ đời thường, tự nhiên (ví dụ: dùng từ "đau bao tử" thay vì "viêm loét dạ dày", "nhức đầu" thay vì "đau nửa đầu đầu").
2. Các câu hỏi phải đa dạng về cách tiếp cận (hỏi triệu chứng, hỏi dấu hiệu sớm, hỏi cách phòng tránh).
3. ĐẦU RA CHỈ TRẢ VỀ DUY NHẤT ĐỊNH DẠNG JSON LÀ MỘT ĐỐI TƯỢNG CÓ PHẦN TỬ "questions" CHỨA MẢNG CÁC CHUỖI, KHÔNG CHÀO HỎI, KHÔNG GIẢI THÍCH.

[VÍ DỤ MẪU]:
- Tên bệnh: Viêm dạ dày
- Output JSON: 
{{
  "questions": [
    "Dạo này bụng em cứ đau âm ỉ vùng trên rốn sau khi ăn là bị gì ạ?",
    "Cho em hỏi hay bị ợ chua với nóng rát cổ họng có phải đau bao tử không?",
    "Làm sao để biết mình bị đau dạ dày sớm vậy dược sĩ?",
    "Em bị viêm loét dạ dày thì có cần kiêng ăn đồ chua hoàn toàn không?",
    "Bị đau dạ dày cấp tính thì dấu hiệu nhận biết như thế nào?"
  ]
}}

[DỮ LIỆU CẦN XỬ LÝ]:
- Tên bệnh: {disease_name}
- Thông tin: {disease_context}

Output JSON:"""

PRODUCT_PROMPT = """Bạn là một chuyên gia giả lập dữ liệu mua sắm dược phẩm. Dựa trên thông tin sản phẩm thuốc dưới đây, hãy đóng vai khách hàng để sinh ra đúng 3 câu hỏi thực tế, mang tính khẩu ngữ đời thường mà họ sẽ dùng khi đi mua thuốc hoặc hỏi tư vấn viên nhà thuốc.

QUY TẮC CỐ ĐỊNH:
1. Sử dụng văn phong mua sắm, hỏi giá, hỏi cách dùng, hoặc hỏi thuốc này trị bệnh gì theo cách nói của người dân.
2. ĐẦU RA CHỈ TRẢ VỀ DUY NHẤT ĐỊNH DẠNG JSON LÀ MỘT ĐỐI TƯỢNG CÓ PHẦN TỬ "questions" CHỨA MẢNG CÁC CHUỖI, KHÔNG CHÀO HỎI, KHÔNG GIẢI THÍCH.

[VÍ DỤ MẪU]:
- Tên sản phẩm: Panadol Extra
- Output JSON: 
{{
  "questions": [
    "Thuốc panadol đỏ này giá bao nhiêu một vỉ vậy shop?",
    "Em bị sốt kèm nhức đầu uống panadol extra được không và uống mấy viên?",
    "Panadol vỉ màu đỏ này có cần bác sĩ kê đơn mới mua được không ạ?"
  ]
}}

[DỮ LIỆU CẦN XỬ LÝ]:
- Tên sản phẩm: {product_name}
- Thông tin sản phẩm: {product_context}

Output JSON:"""

# --- 4. HÀM GỌI API AN TOÀN (Rate Limit + Backoff) ---
def call_gemini_api(prompt, max_retries=10):
    for attempt in range(max_retries):
        try:
            # Nghỉ 4.5 giây giữa các request để đảm bảo dưới giới hạn 15 RPM của Gemini Free Tier
            time.sleep(5.5)
            response = model.generate_content(prompt)
            res_content = response.text
            parsed = json.loads(res_content)
            # Trích xuất mảng câu hỏi
            if isinstance(parsed, dict) and "questions" in parsed:
                return parsed["questions"]
            elif isinstance(parsed, list):
                return parsed
            else:
                # Nếu LLM trả về key khác, thử lấy giá trị mảng đầu tiên tìm thấy
                for val in parsed.values():
                    if isinstance(val, list):
                        return val
                return []
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "rate_limit" in err_msg.lower() or "resource_exhausted" in err_msg.lower():
                # Gemini reset rate limit theo phút. Nghỉ 60 giây khi chạm limit.
                print(f"\n⚠️ Chạm Rate Limit (429). Tự động nghỉ 60 giây để reset giới hạn... (Lần thử {attempt + 1}/{max_retries})")
                time.sleep(60)
            else:
                print(f"\n❌ Lỗi gọi API: {err_msg}")
                time.sleep(5)
    return []

# --- 5. TIỆN ÍCH LƯU TIẾN TRÌNH ---
def load_checkpoint(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Lỗi đọc file checkpoint {file_path}, khởi tạo mới: {e}")
    return {}

def save_checkpoint(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# --- 6. SINH CÂU HỎI CHO BỆNH LÝ ---
def generate_medical_qa(limit=None):
    print("\n=== BƯỚC 2A: SINH CÂU HỎI CHO BỆNH LÝ (MEDICAL) ===")
    if not os.path.exists(MEDICAL_CSV):
        print(f"⚠️ Không tìm thấy file: {MEDICAL_CSV}")
        return {}

    df = pd.read_csv(MEDICAL_CSV).fillna("N/A")
    # Gom nhóm theo category (bệnh)
    groups = df.groupby("category")
    print(f"Tìm thấy {len(groups)} bệnh độc bản trong dữ liệu thô.")

    checkpoint_data = load_checkpoint(MEDICAL_QA_JSON)
    print(f"Đã tải {len(checkpoint_data)} bệnh từ checkpoint trước.")

    count = 0
    for category, group in tqdm(groups, desc="Sinh QA bệnh lý"):
        # Áp dụng limit nếu có
        if limit is not None and count >= limit:
            print(f"Đã đạt giới hạn --limit-med: {limit}")
            break

        # Nếu đã sinh bệnh này rồi thì bỏ qua (checkpoint)
        if category in checkpoint_data:
            continue

        # Lấy tên hiển thị sạch (loại bỏ hậu tố - DinhNghia...)
        display_names = group['display_name'].tolist()
        clean_name = display_names[0].split(" - ")[0].strip() if display_names else str(category)
        
        # Tạo ngữ cảnh chung từ tất cả các dòng của bệnh
        context_parts = []
        for _, row in group.iterrows():
            context_parts.append(str(row.get('text', '')))
        disease_context = "\n\n".join(context_parts)[:3500] # Giới hạn độ dài ngữ cảnh

        prompt = MEDICAL_PROMPT.format(disease_name=clean_name, disease_context=disease_context)
        questions = call_gemini_api(prompt)

        if questions:
            checkpoint_data[category] = {
                "display_name": clean_name,
                "questions": questions
            }
            save_checkpoint(checkpoint_data, MEDICAL_QA_JSON)
            count += 1
            print(f"✅ Đã sinh xong câu hỏi cho bệnh: {clean_name} ({len(checkpoint_data)}/607)")
        else:
            print(f"⚠️ Không sinh được câu hỏi cho bệnh: {clean_name}")

    print(f"✅ Hoàn tất sinh câu hỏi y tế! Tổng cộng có: {len(checkpoint_data)} bệnh đã được xử lý.")
    return checkpoint_data

# --- 7. SINH CÂU HỎI CHO SẢN PHẨM (PRODUCTS) ---
def generate_products_qa(limit=None):
    print("\n=== BƯỚC 2B: SINH CÂU HỎI CHO SẢN PHẨM (PRODUCTS) ===")
    if not os.path.exists(PRODUCTS_CSV):
        print(f"⚠️ Không tìm thấy file: {PRODUCTS_CSV}")
        return {}

    df = pd.read_csv(PRODUCTS_CSV).fillna("N/A")
    # Gom nhóm theo product_name để tránh trùng lặp
    groups = df.groupby("product_name")
    print(f"Tìm thấy {len(groups)} sản phẩm độc bản trong dữ liệu thô.")

    checkpoint_data = load_checkpoint(PRODUCTS_QA_JSON)
    print(f"Đã tải {len(checkpoint_data)} sản phẩm từ checkpoint trước.")

    count = 0
    for product_name, group in tqdm(groups, desc="Sinh QA sản phẩm"):
        if limit is not None and count >= limit:
            print(f"Đã đạt giới hạn --limit-prod: {limit}")
            break

        if product_name in checkpoint_data:
            continue

        row = group.iloc[0]
        product_context = str(row.get('text', ''))[:3500] # Giới hạn độ dài ngữ cảnh tránh chậm/lỗi
        category = str(row.get('category', 'N/A'))
        display_name = str(row.get('display_name', 'N/A')) # Nhóm sản phẩm

        prompt = PRODUCT_PROMPT.format(product_name=product_name, product_context=product_context)
        questions = call_gemini_api(prompt)

        if questions:
            checkpoint_data[product_name] = {
                "category": category,
                "display_name": display_name,
                "questions": questions
            }
            save_checkpoint(checkpoint_data, PRODUCTS_QA_JSON)
            count += 1
            print(f"✅ Đã sinh xong câu hỏi cho sản phẩm: {product_name} ({len(checkpoint_data)}/7509)")
        else:
            print(f"⚠️ Không sinh được câu hỏi cho sản phẩm: {product_name}")

    print(f"✅ Hoàn tất sinh câu hỏi sản phẩm! Tổng cộng có: {len(checkpoint_data)} sản phẩm đã được xử lý.")
    return checkpoint_data

# --- 8. NẠP VÀO VECTOR DATABASE CHROMADB ---
def ingest_qa_to_chroma():
    print("\n=== BƯỚC 3: NẠP DỮ LIỆU CÂU HỎI GIẢ LẬP VÀO CHROMADB ===")
    
    if not os.path.exists(DB_PATH):
        print(f"❌ Không tìm thấy Vector DB cũ tại: {DB_PATH}. Hãy chắc chắn đã chạy s2_ingest_master.py trước.")
        return

    # Khởi tạo client tới DB hiện tại
    client_db = chromadb.PersistentClient(path=DB_PATH)
    
    # Model paraphrase-multilingual hỗ trợ tiếng Việt cực tốt (đồng bộ với s2_ingest_master.py)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Mở collection hiện có
    try:
        collection = client_db.get_collection(
            name="pharmacy_knowledge",
            embedding_function=embedding_func
        )
        print(f"Đã mở collection 'pharmacy_knowledge'. Số bản ghi hiện tại: {collection.count()}")
    except Exception as e:
        print(f"❌ Lỗi mở collection: {e}. Tạo mới...")
        collection = client_db.get_or_create_collection(
            name="pharmacy_knowledge",
            embedding_function=embedding_func,
            metadata={"hnsw:space": "cosine"}
        )

    # 1. Đọc dữ liệu y khoa giả lập
    med_data = load_checkpoint(MEDICAL_QA_JSON)
    # 2. Đọc dữ liệu sản phẩm giả lập
    prod_data = load_checkpoint(PRODUCTS_QA_JSON)

    ids = []
    documents = []
    metadatas = []
    seen_ids = set()

    # Xử lý Medical QA
    for category, info in med_data.items():
        display_name = info.get("display_name", "N/A")
        for q in info.get("questions", []):
            q_clean = str(q).strip()
            if not q_clean:
                continue
            # Logic tạo ID ổn định không trùng lắp (kết hợp category và text)
            hash_input = f"{category}_{q_clean}"
            q_hash = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
            q_id = f"SYN_MED_{q_hash}"

            if q_id in seen_ids:
                continue
            seen_ids.add(q_id)

            # Cú pháp tìm kiếm hybrid (có dấu + không dấu)
            q_no_accent = unidecode(q_clean)
            full_doc = f"{display_name} | {q_clean} | {q_no_accent}"

            ids.append(q_id)
            documents.append(full_doc)
            metadatas.append({
                "original_content": q_clean,
                "type": "synthetic_qa",
                "sub_type": "medical",
                "category": str(category),
                "display_name": display_name,
                "image": "N/A"
            })

    # Xử lý Product QA
    for product_name, info in prod_data.items():
        category = info.get("category", "N/A")
        display_name = info.get("display_name", "N/A")
        for q in info.get("questions", []):
            q_clean = str(q).strip()
            if not q_clean:
                continue
            # Logic tạo ID ổn định không trùng lắp (kết hợp product_name và text)
            hash_input = f"{product_name}_{q_clean}"
            q_hash = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
            q_id = f"SYN_PROD_{q_hash}"

            if q_id in seen_ids:
                continue
            seen_ids.add(q_id)

            q_no_accent = unidecode(q_clean)
            full_doc = f"{product_name} | {q_clean} | {q_no_accent}"

            ids.append(q_id)
            documents.append(full_doc)
            metadatas.append({
                "original_content": q_clean,
                "type": "synthetic_qa",
                "sub_type": "product",
                "category": str(category),
                "product_name": str(product_name),
                "display_name": display_name,
                "image": "N/A"
            })

    total_records = len(ids)
    print(f"Tổng số câu hỏi giả lập chuẩn bị nạp: {total_records} (Medical: {len(med_data)*5}, Product: {len(prod_data)*3} ước tính)")

    if total_records == 0:
        print("⚠️ Không có câu hỏi nào được sinh để nạp vào DB.")
        return

    # Nạp dữ liệu theo Batch 100 bản ghi
    batch_size = 100
    for idx in tqdm(range(0, total_records, batch_size), desc="Đang nạp vào ChromaDB"):
        batch_ids = ids[idx: idx + batch_size]
        batch_docs = documents[idx: idx + batch_size]
        batch_meta = metadatas[idx: idx + batch_size]
        
        # Sử dụng upsert để ghi đè nếu trùng lặp ID (an toàn không bị lỗi trùng ID)
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta
        )

    print(f"✅ HOÀN TẤT NẠP DỮ LIỆU! Số lượng bản ghi trong ChromaDB hiện tại: {collection.count()}")

# --- 9. HÀM CHÍNH ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline sinh dữ liệu giả lập (Synthetic QA) và nạp vào ChromaDB.")
    parser.add_argument("--limit-med", type=int, default=None, help="Giới hạn số lượng bệnh lý cần sinh câu hỏi.")
    parser.add_argument("--limit-prod", type=int, default=None, help="Giới hạn số lượng thuốc/sản phẩm cần sinh câu hỏi.")
    parser.add_argument("--ingest", action="store_true", help="Có thực hiện nạp dữ liệu vào ChromaDB sau khi sinh xong hay không.")
    args = parser.parse_args()

    # Bước 1 & 2: Trích xuất và Sinh câu hỏi qua LLM
    generate_medical_qa(limit=args.limit_med)
    generate_products_qa(limit=args.limit_prod)

    # Bước 3: Nạp vào Vector DB
    if args.ingest:
        ingest_qa_to_chroma()
    else:
        print("\n💡 Gợi ý: Hãy thêm cờ '--ingest' để tự động nạp dữ liệu đã sinh vào ChromaDB.")
