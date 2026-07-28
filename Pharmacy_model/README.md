# Pharmacy_model

## Mục tiêu dự án

`Pharmacy_model` là **hệ thống đối sánh tư vấn dược phẩm** phục vụ việc chuẩn hóa dữ liệu, truy hồi tri thức và đánh giá các hướng tiếp cận mô hình để gợi ý sản phẩm/phác đồ hỗ trợ phù hợp theo nhu cầu người dùng.

Dự án đang được tổ chức theo hướng dữ liệu trung tâm, có các lớp xử lý từ `bronze` → `silver`, sau đó phục vụ cho các mô hình baseline, graph và RAG.

## Directory Tree hiện tại

```text
Pharmacy_model/
├── app/
│   ├── main.py
│   ├── main_v2.py
│   ├── main_navie_rag.py
│   ├── main_advanced_rag.py
│   ├── main_advanced_rag_1.py
│   ├── advanced.py
│   ├── interface.py
│   ├── utils.py
│   └── templates/
│       └── index.html
├── data/
│   ├── bronze/
│   │   └── products/
│   │       ├── longchau_thuoc.json
│   │       ├── longchau_thuoc_full.json
│   │       ├── longchau_duocmypham_full.json
│   │       ├── longchau_thucphamchucnang_full.json
│   │       ├── longchau_thietbiyte_full.json
│   │       └── longchau_chamsoccanhan_full.json
│   ├── silver/
│   │   ├── products_cleaned.csv
│   │   ├── disease_category_map.csv
│   │   ├── evaluation_test.csv
│   │   ├── medical_kb.csv
│   │   ├── products_kb.csv
│   │   ├── graph_edges.csv
│   │   ├── categories_config.json
│   │   ├── extracted_keywords.json
│   │   ├── extracted_keywords_top10.json
│   │   ├── extracted_keywords_top10_strict.json
│   │   ├── synthetic_medical_qa.json
│   │   └── synthetic_products_qa.json
│   └── vector_db/
│       ├── chroma.sqlite3
│       └── e3cb5d1b-3bb6-47e8-a00e-e9607c2e41db/
│           ├── data_level0.bin
│           ├── header.bin
│           ├── index_metadata.pickle
│           ├── length.bin
│           └── link_lists.bin
├── models/
│   ├── medical_classifier.pkl
│   └── medical_classifier_strict.pkl
├── reports/
│   ├── baocao_advanced_rag.md
│   ├── baocao_knowledge_graph.md
│   ├── baocao_ml_baseline_appendix.md
│   ├── baseline_strict_report.txt
│   └── benchmark_tong_hop_3_model.md
├── scripts/
│   ├── build_strict_keywords.py
│   ├── export_graph_edges.py
│   ├── extract_keywords.py
│   ├── s1_append_clean_data.py
│   ├── s1_clean_data_master.py
│   ├── s2_ingest_master.py
│   ├── s2_ingest_to_chroma.py
│   ├── s3_generate_synthetic_qa.py
│   ├── s4_products_clean.py
│   └── archive/
│       ├── ingest_datasets.py
│       ├── ingest_json.py
│       └── ingest_pdf.py
├── src/
│   ├── dynamic_diagnosis.py
│   ├── product_text_utils.py
│   ├── baseline_ml/
│   │   ├── data_processor.py
│   │   ├── inference.py
│   │   └── trainer.py
│   ├── graph_embedding/
│   │   └── graph_model.py
│   └── rag_system/
│       ├── ingest.py
│       └── inference.py
├── tests/
├── requirements.txt
├── requirements_ft.txt
├── app.py
├── config.py
├── evaluate_models.py
├── main_baseline.py
├── pipeline_data_processing.py
├── product_prompt.py
├── run_clinical_validation.py
├── run_graph_inference.py
├── run_rag_inference.py
├── run_rag_ingest.py
└── README.md
```

## Trạng thái hiện tại

### Pipeline 1 — Làm sạch sản phẩm
- **ĐÃ HOÀN THÀNH**
- Đầu ra chính: `data/silver/products_cleaned.csv`

### Pipeline 2 — Đồng bộ nhãn bệnh
- **ĐÃ HOÀN THÀNH**
- Đầu ra chính: `data/silver/disease_category_map.csv`
- Ghi chú: file này đã được **gán nhãn thủ công** để dùng làm nền cho benchmark và đánh giá mô hình

## Kế hoạch tiếp theo

Chuẩn bị phát triển thư mục `src/benchmark/` để chuẩn hóa 3 hướng đối sánh và 1 bộ đánh giá chung:

```text
src/benchmark/
├── ml_baseline/
│   └── ...
├── graph_db/
│   └── ...
├── rag_pure/
│   └── ...
└── evaluator.py
```

### Mục tiêu của `src/benchmark/`
- **ML Baseline**: mô hình nền tảng để so sánh tốc độ và độ chính xác cơ bản
- **Graph DB**: truy vấn/đối sánh dựa trên cấu trúc tri thức và quan hệ
- **RAG thuần túy**: truy hồi + sinh câu trả lời trên dữ liệu đã chuẩn hóa
- **Evaluator chung**: thống nhất metric, input/output và format báo cáo cho benchmark

## Ghi chú chuẩn hóa cấu trúc

- Giữ nguyên `venv/` theo chuẩn môi trường cài đặt cục bộ.
- Các file cache sinh ra khi chạy Python đã được dọn ở phạm vi project ngoài `venv/` để tránh nhiễu cấu trúc.
- Khi bắt đầu benchmark, nên ưu tiên tách rõ:
  - dữ liệu chuẩn bị (`data/silver/`)
  - mô hình (`src/benchmark/`)
  - kết quả đánh giá (`reports/` hoặc `benchmark_results/`)

## Tóm tắt nhanh

- Dự án đã có đầy đủ nền tảng dữ liệu sạch và mapping nhãn.
- Sẵn sàng chuyển sang giai đoạn benchmark hóa 3 mô hình đối sánh.
- README này đóng vai trò như bản đồ hiện trạng để tiếp tục mở rộng kiến trúc một cách có kiểm soát.
