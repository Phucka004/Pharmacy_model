import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
import shutil
from tqdm import tqdm
from unidecode import unidecode

def ingest_to_chroma():
    medical_csv = "data/silver/medical_kb.csv"
    products_csv = "data/silver/products_kb.csv"
    db_path = "data/vector_db"
    
    # 1. Dọn dẹp DB cũ
    if os.path.exists(db_path):
        print(f"🧹 Đang dọn sạch Vector DB cũ tại {db_path}...")
        shutil.rmtree(db_path)

    # 2. Khởi tạo Client
    client = chromadb.PersistentClient(path=db_path)
    
    # Model paraphrase-multilingual hỗ trợ tiếng Việt cực tốt
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    collection = client.get_or_create_collection(
        name="pharmacy_knowledge",
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"}
    )

    def process_and_ingest(file_path, is_product_file=False):
        if not os.path.exists(file_path):
            print(f"⚠️ Không tìm thấy: {file_path}")
            return
        
        # Đọc dữ liệu
        df = pd.read_csv(file_path).fillna("N/A")
        total_rows = len(df)
        prefix_id = "PROD" if is_product_file else "MED"
        
        print(f"\n🚀 Đang nạp {total_rows} bản ghi từ {os.path.basename(file_path)}...")

        batch_size = 50 
        
        for i in tqdm(range(0, total_rows, batch_size), desc=f"Nạp {prefix_id}"):
            batch = df.iloc[i : i + batch_size]
            
            ids = []
            documents = []
            metadatas = []
            
            for index, row in batch.iterrows():
                content = str(row.get('text', ''))
                d_name = str(row.get('display_name', 'N/A'))
                cat = str(row.get('category', 'N/A'))
                p_name = str(row.get('product_name', 'N/A'))
                
                # --- XỬ LÝ DẤU THÔNG MINH ---
                # Tạo bản không dấu để hỗ trợ search (Hybrid Indexing)
                content_no_accent = unidecode(content)
                # Ghép tất cả vào để Vector DB dễ bắt được từ khóa
                full_doc_for_indexing = f"{d_name} | {content} | {content_no_accent}"
                
                ids.append(f"{prefix_id}_{i + index}")
                documents.append(full_doc_for_indexing)
                
                # Metadata lưu bản gốc (Original Content) để LLM đọc cho chuẩn
                if is_product_file:
                    metadatas.append({
                        "original_content": content,
                        "type": "product",
                        "source": str(row.get('source_group', 'N/A')),
                        "category": cat,
                        "product_name": p_name,
                        "display_name": d_name,
                        "price": str(row.get('sale_price', 'N/A')),
                        "is_etc": str(row.get('is_prescription_label', 'Không')),
                        "image": str(row.get('image_url', 'N/A'))
                    })
                else:
                    metadatas.append({
                        "original_content": content,
                        "type": "medical",
                        "source": str(row.get('source', 'N/A')),
                        "category": cat,
                        "display_name": d_name,
                        "image": "N/A"
                    })

            # Đẩy vào ChromaDB theo từng batch
            if ids:
                collection.add(ids=ids, documents=documents, metadatas=metadatas)
            
    # Chạy nạp dữ liệu
    process_and_ingest(medical_csv, is_product_file=False)
    process_and_ingest(products_csv, is_product_file=True)

    print(f"\n✅ HOÀN TẤT! Vector DB hiện có: {collection.count()} bản ghi.")

if __name__ == "__main__":
    ingest_to_chroma()