import json
import pandas as pd
import os
import random
import unicodedata
import re
from tqdm import tqdm

# --- Tái sử dụng các hàm làm sạch từ code gốc của ông giáo ---
def normalize_for_matching(s):
    if not s: return ""
    s = s.lower()
    dấu = {'a': 'àáạảãâầấậẩẫăằắặẳẵ', 'e': 'èéẹẻẽêềếệểễ', 'i': 'ìíịỉĩ', 'o': 'òóọỏõôồốộổỗơờớợởỡ', 'u': 'ùúụủũưừứựửữ', 'y': 'ỳýỵỷỹ', 'd': 'đ'}
    for char, group in dấu.items():
        for g in group: s = s.replace(g, char)
    return re.sub(r'[^a-z0-9]', '', s)

def process_single_longchau_file(file_path):
    processed = []
    if not os.path.exists(file_path): 
        print(f"❌ Không tìm thấy file: {file_path}")
        return processed
    
    # Xác định nhóm sản phẩm dựa trên tên file
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
                # Logic xử lý giá và thuốc kê đơn (giữ nguyên từ code gốc)
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

                p['sale_price'] = sale_price
                p['is_prescription_label'] = "Có" if is_etc else "Không"
                p['category'] = master_category
                p['display_name'] = sub_category_name
                p['source_group'] = source_label
                
                # Trường text tổng hợp cho RAG
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

def update_missing_file():
    # 1. Đường dẫn file cần bổ sung và file đích
    missing_file = "data/bronze/products/longchau_thuoc.json" 
    output_kb = "data/silver/products_kb.csv"
    
    print(f"🚀 Bắt đầu xử lý file bổ sung: {missing_file}")
    
    # 2. Xử lý làm sạch file mới
    new_products = process_single_longchau_file(missing_file)
    
    if not new_products:
        print("Empty data hoặc không tìm thấy file.")
        return

    df_new = pd.DataFrame(new_products)

    # 3. Ghi vào file đích
    if os.path.exists(output_kb):
        # Nếu file đã tồn tại, dùng mode='a' (append) và không ghi header
        df_new.to_csv(output_kb, mode='a', index=False, header=False, encoding='utf-8-sig')
        print(f"✅ Đã ghi nối {len(new_products)} sản phẩm vào {output_kb}")
    else:
        # Nếu file chưa tồn tại (chưa chạy master_clean trước đó), tạo mới luôn
        df_new.to_csv(output_kb, index=False, encoding='utf-8-sig')
        print(f"✅ Đã tạo mới và lưu {len(new_products)} sản phẩm vào {output_kb}")

    # 4. (Tùy chọn) Lọc trùng sau khi gom để đảm bảo data sạch tuyệt đối
    df_final = pd.read_csv(output_kb)
    initial_count = len(df_final)
    df_final = df_final.drop_duplicates(subset=['text'])
    df_final.to_csv(output_kb, index=False, encoding='utf-8-sig')
    
    print(f"📊 Tổng sản phẩm hiện tại: {len(df_final)} (Đã loại bỏ {initial_count - len(df_final)} trùng lặp)")
    print("="*30)

if __name__ == "__main__":
    update_missing_file()