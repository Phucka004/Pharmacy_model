import os
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def ingest_all_csv():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    data_folder = "data/datasets" # Để tất cả file CSV vào đây
    all_docs = []

    for file_name in os.listdir(data_folder):
        if file_name.endswith(".csv"):
            print(f" sedang xử lý {file_name}...")
            df = pd.read_csv(os.path.join(data_folder, file_name))
            
            # Tự động nhận diện cột (Nếu là ViMedical thì lấy Disease và Question)
            for _, row in df.iterrows():
                # Gộp tất cả các cột thành 1 chuỗi văn bản để AI tìm kiếm
                content = " ".join([str(v) for v in row.values])
                all_docs.append(Document(page_content=content, metadata={"source": file_name}))

    # Nạp một lần duy nhất vào ngăn bệnh
    vector_db = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory="chroma_db",
        collection_name="disease_knowledge"
    )
    print("✅ HOÀN TẤT INDEXING DATASETS")