import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
import shutil

def ingest_to_chroma():
    csv_path = "data/silver/knowledge_base.csv"
    db_path = "data/vector_db"
    
    # BẮT BUỘC: Xóa DB cũ vì cấu trúc Vector đã thay đổi hoàn toàn
    if os.path.exists(db_path):
        print(f"--- Dọn dẹp Vector DB cũ (Cấu trúc không dấu cũ) ---")
        shutil.rmtree(db_path)

    client = chromadb.PersistentClient(path=db_path)
    
    # Model này cực mạnh cho tiếng Việt có dấu
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    collection = client.get_or_create_collection(
        name="pharmacy_knowledge",
        embedding_function=embedding_func
    )

    df = pd.read_csv(csv_path).fillna("N/A")
    total = len(df)
    
    print(f"--- Bắt đầu nạp {total} bản ghi CÓ DẤU vào ChromaDB ---")
    
    batch_size = 200 # Giảm batch size để ổn định hơn cho model Transformer
    for i in range(0, total, batch_size):
        batch = df.iloc[i : i + batch_size]
        
        ids = [f"id_{j}" for j in range(i, i + len(batch))]
        
        # Gia cố Document bằng tên bệnh để tăng trọng số truy xuất
        documents = []
        metadatas = []
        
        for _, row in batch.iterrows():
            d_name = str(row.get('disease', 'N/A'))
            t_content = str(row.get('text', ''))
            
            # Gắn nhãn bệnh trực tiếp vào văn bản để Vector "ám ảnh" tên bệnh
            enriched_text = f"BỆNH: {d_name} | NỘI DUNG: {t_content}"
            documents.append(enriched_text)
            
            # Metadata chi tiết để lọc (Filtering) ở bước Retrieval
            metadatas.append({
                "disease": d_name,
                "source": str(row.get('source', 'Unknown')),
                "sku": str(row.get('sku', 'N/A'))
            })

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"-> Đã nạp: {min(i + batch_size, total)}/{total}")

    print("--- HOÀN TẤT! Hệ thống đã nạp xong dữ liệu Tiếng Việt chuẩn Advanced ---")

if __name__ == "__main__":
    ingest_to_chroma()