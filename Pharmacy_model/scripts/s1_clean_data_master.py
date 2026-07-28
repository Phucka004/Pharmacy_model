import json
import pandas as pd
import os
import re
import glob
import random
import unicodedata
from bs4 import BeautifulSoup
from groq import Groq
from tqdm import tqdm
import time

# --- 1. CẤU HÌNH GROQ (SỬ DỤNG MODEL MỚI NHẤT) ---
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def call_llm_summarize(disease_name, context):
    if not context or len(context) < 100:
        return "Dữ liệu quá ngắn."
    
    prompt = f"""Bạn là dược sĩ chuyên gia. Hãy tóm tắt bệnh "{disease_name}" theo cấu trúc JSON sau:
    {{
      "DinhNghia": "...",
      "NguyenNhan": "...",
      "ChanDoan": "...",
      "CachChua": "...",
      "PhongNgua": "..."
    }}
    Văn bản: {context[:4000]}
    Chỉ trả ra JSON, không giải thích."""
    
    try:
        time.sleep(0.5) 
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant", 
            temperature=0.1,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Lỗi tóm tắt tại {disease_name}: {str(e)}"

# --- 2. CÁC HÀM LÀM SẠCH ---
def super_clean_medical_text(text, is_question=False):
    if not text: return ""
    text = unicodedata.normalize('NFC', text)
    trash_patterns = [
        r"Trang chủ\s*>\s*.*?>", r"Mục lục", r"Nguồn ảnh:.*?\n",
        r"Hệ thống Bệnh viện Đa khoa Tâm Anh", r"https?://\S+", r"www\.\S+"
    ]
    if is_question:
        q_prefixes = [r"Tôi đang bị\s*", r"Cho hỏi\s*", r"Tôi có triệu chứng\s*"]
        trash_patterns.extend(q_prefixes)
    for p in trash_patterns: text = re.sub(p, '', text, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r'\s+', ' ', text).strip()

def normalize_for_matching(s):
    if not s: return ""
    s = s.lower()
    dấu = {'a': 'àáạảãâầấậẩẫăằắặẳẵ', 'e': 'èéẹẻẽêềếệểễ', 'i': 'ìíịỉĩ', 'o': 'òóọỏõôồốộổỗơờớợởỡ', 'u': 'ùúụủũưừứựửữ', 'y': 'ỳýỵỷỹ', 'd': 'đ'}
    for char, group in dấu.items():
        for g in group: s = s.replace(g, char)
    return re.sub(r'[^a-z0-9]', '', s)

# --- 3. XỬ LÝ VIMQ ---
def process_vimq(folder_path):
    processed = []
    files = ['train.json', 'test.json', 'dev.json']
    for file_name in files:
        path = os.path.join(folder_path, file_name)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    label = item.get('sent_label', 'y tế')
                    processed.append({
                        "text": f"### [MẪU CÂU HỎI {label.upper()}]\nCâu hỏi: {item['sentence']}",
                        "source": f"ViMQ_{file_name}",
                        "category": f"INTENT:{normalize_for_matching(label)}",
                        "display_name": "Tình huống thực tế"
                    })
    return processed

# --- 4. XỬ LÝ VIMEDICAL ---
def process_vimedical_comprehensive(base_path):
    kb_data = []
    test_queries = []
    symptom_path = os.path.join(base_path, "Corpus_Redone")
    question_path = os.path.join(base_path, "Question_for_dataset")
    html_path = os.path.join(base_path, "Corpus")

    if not os.path.exists(symptom_path):
        print(f"⚠️ Không tìm thấy đường dẫn: {symptom_path}")
        return kb_data, test_queries

    q_map = {normalize_for_matching(f.replace(".txt", "")): f for f in os.listdir(question_path) if f.endswith('.txt')}
    disease_files = [f for f in os.listdir(symptom_path) if f.endswith('.txt')]
    
    print(f"🔍 Tìm thấy {len(disease_files)} bệnh. Bắt đầu xử lý...")

    for filename in tqdm(disease_files, desc="Đang xử lý dữ liệu bệnh"):
        raw_key = filename.replace(".txt", "")
        match_key = normalize_for_matching(raw_key)
        master_category = f"BENH:{match_key}"
        display_name = raw_key.replace("-", " ").title()

        # A. Triệu chứng
        with open(os.path.join(symptom_path, filename), 'r', encoding='utf-8') as f:
            content = super_clean_medical_text(f.read())
            kb_data.append({"text": f"### [TRIỆU CHỨNG: {display_name}]\n{content}", "source": "ViMedical_Base", "category": master_category, "display_name": display_name})

        # B. Test Queries (Lấy đúng 5 câu mỗi file)
        q_file = q_map.get(match_key)
        if q_file:
            with open(os.path.join(question_path, q_file), 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f.readlines() if len(l.strip()) > 10]
                for q in lines[:5]:
                    test_queries.append({"question": super_clean_medical_text(q, True), "expected": display_name})

        # C. Tóm tắt HTML bằng LLM
        h_file = os.path.join(html_path, raw_key + ".html")
        if os.path.exists(h_file):
            with open(h_file, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                for tag in soup(["script", "style"]): tag.decompose()
                deep_raw = super_clean_medical_text(soup.get_text())
                
                # Gọi LLM và bóc tách JSON
                summary_raw = call_llm_summarize(display_name, deep_raw[:3000])
                if summary_raw:
                    try:
                        # Tìm phần chứa JSON (phòng hờ LLM trả lời thừa text)
                        json_match = re.search(r'\{.*\}', summary_raw, re.DOTALL)
                        if json_match:
                            summary_dict = json.loads(json_match.group())
                            for title, content in summary_dict.items():
                                if content and len(str(content)) > 10:
                                    kb_data.append({
                                        "text": f"### [{title.upper()}: {display_name}]\n{content.strip()}",
                                        "source": "ViMedical_Summary",
                                        "category": master_category, 
                                        "display_name": f"{display_name} - {title}" # Phân loại rõ để search
                                    })
                    except:
                        pass # Nếu lỗi JSON thì bỏ qua khâu tóm tắt bệnh đó
    
    return kb_data, test_queries

# --- 5. XỬ LÝ LONG CHÂU (GIỮ HẾT THUỘC TÍNH + FIX GIÁ ETC) ---
def process_longchau(file_path):
    processed = []
    if not os.path.exists(file_path): return processed
    
    filename_lower = os.path.basename(file_path).lower()
    source_map = {
        "thietbiyte": "THIẾT BỊ Y TẾ", "thucphamchucnang": "THỰC PHẨM CHỨC NĂNG",
        "duocmypham": "DƯỢC MỸ PHẨM", "chamsoccanhan": "CHĂM SÓC CÁ NHÂN", "thuoc": "THUỐC ĐIỀU TRỊ"
    }
    source_label = next((v for k, v in source_map.items() if k in filename_lower), "SẢN PHẨM KHÁC")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for cat_slug, products in data.items():
            master_category = f"SP:{normalize_for_matching(cat_slug)}"
            sub_category_name = cat_slug.replace("-", " ").title()
            
            for p in products:
                # Logic xử lý giá và thuốc kê đơn
                sale_price = p.get('sale_price')
                is_etc = p.get('is_prescription') is True or "Kê đơn" in str(p.get('is_prescription', ''))
                
                if is_etc:
                    if not sale_price or str(sale_price).upper() == "N/A":
                        fake_price = random.randint(100, 800) * 1000
                        sale_price = f"{fake_price:,}đ (Giá tham khảo - Cần đơn thuốc)".replace(",", ".")
                    else:
                        sale_price = f"{sale_price} (Cần đơn thuốc)"
                elif not sale_price or str(sale_price).upper() == "N/A":
                    fake_price = random.randint(20, 300) * 1000
                    sale_price = f"{fake_price:,}đ (Giá tham khảo)".replace(",", ".")

                # Cập nhật thông tin vào object p
                p['sale_price'] = sale_price
                p['is_prescription_label'] = "Có" if is_etc else "Không"
                p['category'] = master_category
                p['display_name'] = sub_category_name
                p['source_group'] = source_label
                
                # Tạo trường text tổng hợp cho RAG
                p['text'] = (
                    f"### [SẢN PHẨM: {p.get('product_name', 'N/A')}]\n"
                    f"- Nhóm: {source_label}\n"
                    f"- Giá: {sale_price}\n"
                    f"- Thuốc kê đơn: {p['is_prescription_label']}\n"
                    f"- Công dụng: {p.get('usage', 'N/A')}\n"
                    f"- Thành phần: {p.get('ingredients_raw', 'N/A')}\n"
                    f"- Cách dùng: {p.get('dosage', 'N/A')}"
                )
                processed.append(p)
    return processed

# --- 6. MASTER RUN ---
def master_clean():
    os.makedirs("data/silver", exist_ok=True)
    
    # 1. Xử lý Bệnh (ViMedical)
    med_kb, med_test = process_vimedical_comprehensive("data/bronze/ViMedical_Disease/RAW DATA")
    
    # 2. Xử lý Ý định (ViMQ)
    vimq_data = process_vimq("data/bronze/ViMQ")
    med_kb.extend(vimq_data)
    
    # 3. Xử lý Sản phẩm (Long Châu)
    all_products = []
    # Tìm tất cả file json của Long Châu trong bronze
    product_files = glob.glob("data/bronze/**/longchau_*.json", recursive=True)
    print(f"📦 Tìm thấy {len(product_files)} file sản phẩm Long Châu.")
    for f in product_files:
        all_products.extend(process_longchau(f))
    
    # Xuất dữ liệu
    if med_kb:
        pd.DataFrame(med_kb).to_csv("data/silver/medical_kb.csv", index=False, encoding='utf-8-sig')
    if all_products:
        pd.DataFrame(all_products).to_csv("data/silver/products_kb.csv", index=False, encoding='utf-8-sig')
    if med_test:
        pd.DataFrame(med_test).to_csv("data/silver/evaluation_test.csv", index=False, encoding='utf-8-sig')
        
    print("\n" + "="*30)
    print("✅ HOÀN TẤT QUÁ TRÌNH LÀM SẠCH!")
    print(f"- Kiến thức y khoa: {len(med_kb)} bản ghi")
    print(f"- Sản phẩm Long Châu: {len(all_products)} bản ghi")
    print(f"- Bộ câu hỏi test: {len(med_test)} câu")
    print("="*30)

if __name__ == "__main__":
    master_clean()