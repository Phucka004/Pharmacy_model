````markdown
# 💊 Pharmacy_model

Hệ thống ứng dụng AI để **dự đoán bệnh dựa trên triệu chứng** và **gợi ý sản phẩm thuốc không kê đơn (OTC)** bằng **Machine Learning, Knowledge Graph và Advanced Retrieval-Augmented Generation (RAG)**.

> **Dữ liệu hỏi đáp y tế (Medical QA) được điều chỉnh từ bộ dữ liệu [ViMedical_Disease](https://github.com/PB3002/ViMedical_Disease).**

---

# Giới thiệu

`Pharmacy_model` là dự án nghiên cứu nhằm xây dựng và đánh giá các mô hình AI cho bài toán:

- Dự đoán bệnh từ triệu chứng người dùng.
- Gợi ý sản phẩm thuốc không kê đơn (OTC).
- So sánh hiệu quả giữa nhiều hướng tiếp cận AI trên cùng một bộ dữ liệu và tiêu chí đánh giá.

Dự án được thiết kế theo hướng **Data-Centric AI**, bao gồm toàn bộ quy trình từ thu thập dữ liệu, tiền xử lý, xây dựng cơ sở tri thức đến đánh giá mô hình.

---

# Kiến trúc dự án

```text
Pharmacy_model
│
├── app/                     # Giao diện Web
├── data/
│   ├── bronze/              # Dữ liệu thô
│   ├── silver/              # Dữ liệu đã chuẩn hóa
│   └── vector_db/           # Chroma Vector Database
│
├── models/                  # Mô hình Machine Learning
├── reports/                 # Báo cáo benchmark
├── scripts/                 # Pipeline xử lý dữ liệu
├── src/
│   ├── benchmark/           # Benchmark các mô hình
│   ├── baseline_ml/
│   ├── graph_embedding/
│   └── rag_system/
│
├── tests/
├── requirements.txt
└── README.md
```

---

# Nguồn dữ liệu

## 1. Dữ liệu hỏi đáp y tế

Dữ liệu Medical Question Answer được điều chỉnh từ bộ dữ liệu:

> https://github.com/PB3002/ViMedical_Disease

---

## 2. Dữ liệu sản phẩm

Thông tin thuốc và các sản phẩm chăm sóc sức khỏe được thu thập từ website **Nhà thuốc Long Châu** nhằm phục vụ mục đích nghiên cứu.

Sau khi thu thập, dữ liệu được:

- Làm sạch
- Chuẩn hóa
- Chuẩn hóa danh mục
- Trích xuất thành phần
- Xây dựng cơ sở tri thức

để phục vụ cho quá trình huấn luyện và truy hồi thông tin.

---

# Pipeline xử lý dữ liệu

```text
Dữ liệu thô
      │
      ▼
Bronze Dataset
      │
      ▼
Làm sạch & Chuẩn hóa
      │
      ▼
Silver Dataset
      │
      ├────────► Machine Learning
      │
      ├────────► Knowledge Graph
      │
      └────────► Advanced RAG
                    │
                    ▼
             Đánh giá mô hình
```

---

# Các mô hình được triển khai

## 1. Machine Learning Baseline

Mô hình nền dùng để so sánh hiệu quả.

Sử dụng:

- TF-IDF
- LinearSVC

Đặc điểm:

- Huấn luyện nhanh
- Suy luận nhanh
- Làm baseline để đánh giá các phương pháp khác

---

## 2. Knowledge Graph

Xây dựng đồ thị tri thức giữa:

- Bệnh
- Triệu chứng
- Thành phần thuốc
- Danh mục thuốc
- Sản phẩm

Giúp cải thiện khả năng truy hồi thông tin dựa trên quan hệ giữa các thực thể.

---

## 3. Advanced RAG

Pipeline RAG gồm nhiều bước:

- Query Expansion
- Hybrid Retrieval
- Embedding Search
- Keyword Search
- Reranking
- Safety Layer
- LLM hỗ trợ chẩn đoán
- Gợi ý sản phẩm phù hợp

---

# Các giai đoạn của dự án

## Giai đoạn 1 — Chuẩn hóa dữ liệu

- Làm sạch dữ liệu sản phẩm
- Chuẩn hóa thông tin

Đầu ra:

```
data/silver/products_cleaned.csv
```

---

## Giai đoạn 2 — Đồng bộ nhãn bệnh

- Chuẩn hóa danh mục bệnh
- Gán nhãn thủ công để phục vụ benchmark

Đầu ra:

```
data/silver/disease_category_map.csv
```

---

## Giai đoạn 3 — Sinh dữ liệu Medical QA

Tạo dữ liệu hỏi đáp y tế phục vụ huấn luyện và đánh giá.

Đầu ra:

```
synthetic_medical_qa.json
```

---

## Giai đoạn 4 — Xây dựng cơ sở tri thức

Bao gồm:

- Medical Knowledge Base
- Product Knowledge Base
- Graph Database
- Chroma Vector Database

---

## Giai đoạn 5 — Đánh giá mô hình

Các mô hình được đánh giá trên cùng tập dữ liệu với các tiêu chí:

- Accuracy
- Precision
- Recall
- F1-score
- Latency

---

# Công nghệ sử dụng

- Python
- Scikit-learn
- ChromaDB
- Sentence Transformers
- NetworkX
- Google Gemini
- Groq
- Ollama
- Pandas
- BeautifulSoup

---
# Mục tiêu nghiên cứu

Dự án hướng tới việc xây dựng một **framework benchmark thống nhất** để so sánh các phương pháp:

- Machine Learning
- Knowledge Graph
- Advanced RAG

trên cùng một bộ dữ liệu và cùng hệ thống đánh giá, từ đó phân tích ưu điểm, hạn chế và khả năng ứng dụng của từng phương pháp trong bài toán tư vấn dược phẩm.

---

# Lưu ý

- Dự án chỉ phục vụ **mục đích học tập và nghiên cứu**.
- Dữ liệu sản phẩm được thu thập từ website **Nhà thuốc Long Châu** nhằm phục vụ nghiên cứu, **không sử dụng cho mục đích thương mại**.
- Hệ thống chỉ mang tính chất tham khảo, **không thay thế ý kiến của bác sĩ hoặc chuyên gia y tế**.
````
