import json
import os
import shutil
import unicodedata
import re
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# THÊM: Tên ngăn chứa cho dữ liệu thuốc tại tiệm
COLLECTION_NAME = "pharmacy_inventory" 
PERSIST_DIR = "chroma_db"

def clean_text(value):
    """Xử lý triệt để ký tự lạ và chuẩn hóa Unicode."""
    if value is None or str(value).strip().lower() in ["n/a", "none", "null", ""]: 
        return "Không rõ"
    
    # Chuyển về string, loại bỏ BOM và ký tự điều khiển
    text = str(value).replace("\ufeff", "").replace("\r", " ").replace("\n", " ")
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\t")
    
    # Làm sạch khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text)
    return unicodedata.normalize("NFC", text).strip()

def record_to_metadata(item):
    """Trích xuất ĐẦY ĐỦ 20+ trường thông tin vào Metadata."""
    fields = [
        "product_name", "sku", "is_prescription", "brand", "brand_origin", 
        "producer", "country", "sale_price", "expiry_date", "unit", 
        "packing", "dosage_form", "regist_num", "ingredients_raw", 
        "description", "usage", "dosage", "side_effects", "precautions", "storage", "expiry_date", "warranty_period" 
    ]
    
    meta = {f: clean_text(item.get(f, "Không rõ")) for f in fields}
    
    # Fallback dự phòng cho lỗi chính tả trường 'precaution' trong JSON
    if meta["precautions"] == "Không rõ" and "precaution" in item:
        meta["precautions"] = clean_text(item.get("precaution"))
        
    return meta

def ingest_all_jsons():
    documents = []
    data_dir = "data"
    records_by_sku = {}

    if not os.path.exists(data_dir):
        print(f"❌ Thư mục {data_dir} không tồn tại!")
        return

    json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    print(f"📂 Khởi động Ingest: Tìm thấy {len(json_files)} file nguồn.")

    for file_name in json_files:
        path = os.path.join(data_dir, file_name)
        count_before = len(records_by_sku)
        try:
            # Dùng utf-8-sig để xử lý file có BOM, errors='replace' để tránh crash do ký tự lạ
            with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
                content = json.load(f)
                
                def find_products(data):
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "product_name" in item:
                                sku = clean_text(item.get("sku", ""))
                                if sku and sku != "Không rõ":
                                    records_by_sku[sku] = item
                    elif isinstance(data, dict):
                        for val in data.values():
                            find_products(val)

                find_products(content)
                print(f"  + [{file_name}]: Đã quét được {len(records_by_sku) - count_before} sản phẩm.")

        except Exception as e:
            print(f"❌ Lỗi xử lý tại file {file_name}: {e}")

    # Chuyển đổi thành Document với nội dung tìm kiếm (content_text) siêu chi tiết
    for sku, item in records_by_sku.items():
        meta = record_to_metadata(item)
        p_name = meta['product_name']
        
        # Cấu trúc này giúp Embedding Model hiểu rõ mối quan hệ giữa các thực thể
        content_text = (
            f"SẢN PHẨM: {p_name}\n"
            f"Mã định danh SKU: {sku}\n"
            f"Giá bán niêm yết: {meta['sale_price']}\n"
            f"Hạn sử dụng (Expiry Date): {meta['expiry_date']}\n"
            f"Thời gian bảo hành (Warranty): {meta['warranty_period']}\n"
            f"Nhà sản xuất: {meta['producer']} tại {meta['country']}\n"
            f"Thương hiệu: {meta['brand']} (Xuất xứ thương hiệu: {meta['brand_origin']})\n"
            f"Dạng bào chế: {meta['dosage_form']}\n"
            f"Quy cách đóng gói: {meta['packing']}\n"
            f"Loại thuốc: {meta['is_prescription']}\n"
            f"Đơn vị tính: {meta['unit']}\n"
            f"Số đăng ký (Regist Num): {meta['regist_num']}\n"
            f"Thành phần chi tiết: {meta['ingredients_raw']}\n"
            f"Công dụng & Chỉ định: {meta['usage']}\n"
            f"Cách dùng & Liều lượng: {meta['dosage']}\n"
            f"Tác dụng phụ: {meta['side_effects']}\n"
            f"Thận trọng & Lưu ý: {meta['precautions']}\n"
            f"Bảo quản: {meta['storage']}\n"
            f"Mô tả tóm tắt: {meta['description']}"
        )
        
        documents.append(Document(page_content=content_text, metadata=meta))

    if not documents:
        print("❌ Không có dữ liệu hợp lệ để nạp vào DB!")
        return

    # Làm sạch database cũ
    if os.path.exists("chroma_db"):
        print("🗑️ Đang xóa database cũ để nạp mới toàn bộ metadata...")
        shutil.rmtree("chroma_db")
    
    print(f"🚀 Đang tạo Vector DB vào ngăn '{COLLECTION_NAME}'...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # CẬP NHẬT: Thêm tham số collection_name vào đây
    Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME
    )
    
    print("="*60)
    print("✅ HOÀN TẤT INDEXING THUỐC!")
    print(f"📊 Đã nạp thành công {len(documents)} sản phẩm vào ngăn '{COLLECTION_NAME}'.")
    print("="*60)

if __name__ == "__main__":
    ingest_all_jsons()