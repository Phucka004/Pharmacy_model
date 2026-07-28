import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def ingest_pdfs():
    # Danh sách file PDF thực tế
    pdf_files = ["data/duoc_thu_quoc_gia_2017.pdf", "data/duoc_thu_quoc_gia_2018.pdf"]
    documents = []
    
    # Bộ cắt văn bản: 1000 ký tự là hợp lý cho Dược thư
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for pdf_path in pdf_files:
        if os.path.exists(pdf_path):
            print(f"📖 Đang đọc: {pdf_path}")
            loader = PyMuPDFLoader(pdf_path)
            pages = loader.load()
            
            # Gán metadata chi tiết TRƯỚC khi split để không mất thông tin trang
            for page in pages:
                year = "2017" if "2017" in pdf_path else "2018"
                page.metadata["source"] = f"Dược thư Quốc gia Việt Nam {year}"
                # PyMuPDFLoader mặc định đã có metadata['page'], ta giữ nguyên nó
            
            chunks = text_splitter.split_documents(pages)
            documents.extend(chunks)

    # Khởi tạo Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Lưu vào ChromaDB
    Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory="chroma_db"
    )
    print(f"✅ Đã nạp {len(documents)} đoạn văn bản từ PDF vào hệ thống.")

if __name__ == "__main__":
    ingest_pdfs()