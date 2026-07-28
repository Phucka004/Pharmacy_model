# Báo cáo tổng kết triển khai Synthetic Data (Synthetic QA Pipeline)

Hệ thống đã xây dựng và kiểm thử thành công pipeline tự động hóa sinh dữ liệu câu hỏi giả lập (Synthetic QA) đời thường từ tập dữ liệu thô bệnh lý (`medical_kb.csv`) và sản phẩm (`products_kb.csv`).

## Kết quả đạt được

1. **Script tự động hóa hoàn chỉnh:** Tạo thành công [s3_generate_synthetic_qa.py](file:///d:/TLCN/Pharmacy_RAG/scripts/s3_generate_synthetic_qa.py) tích hợp toàn bộ các tính năng:
   - Gom nhóm bệnh lý/sản phẩm và tự động trích xuất ngữ cảnh thô (tối ưu hóa độ dài ngữ cảnh tối đa 3500 ký tự để đẩy nhanh tốc độ sinh của LLM và tiết kiệm token).
   - Gọi API Groq (`llama-3.1-8b-instant`) với cấu hình `json_object` chống lỗi định dạng.
   - Cơ chế lưu tiến trình (Checkpointing) ghi nhận dữ liệu lập tức sau mỗi lượt gọi LLM giúp dễ dàng tạm dừng/chạy tiếp mà không tốn chi phí gọi trùng lặp.
   - Cơ chế tự động nghỉ (exponential backoff) thông minh khi chạm Rate Limit (HTTP 429).
   - Nạp dữ liệu (Ingestion) hybrid (chữ có dấu + không dấu thông qua `unidecode`) vào ChromaDB hiện có mà không làm mất dữ liệu hàn lâm cũ.
   - Đảm bảo tính lặp lại (Idempotency) bằng cách sinh ID ổn định dựa trên mã băm MD5 của câu hỏi (`SYN_MED_{hash}` và `SYN_PROD_{hash}`). Điều này giúp chạy nạp nhiều lần mà không tạo ra bản ghi rác bị trùng lặp trong vector DB.

2. **Kết quả thử nghiệm quy mô nhỏ thành công:**
   - **Sinh câu hỏi:** Sinh thành công câu hỏi giả lập cho 6 bệnh lý và 4 sản phẩm.
   - **Định dạng file lưu trữ:** Lưu tại [synthetic_medical_qa.json](file:///d:/TLCN/Pharmacy_RAG/data/silver/synthetic_medical_qa.json) và [synthetic_products_qa.json](file:///d:/TLCN/Pharmacy_RAG/data/silver/synthetic_products_qa.json).
   - **Đồng bộ cơ sở dữ liệu:** Nạp thành công **42 câu hỏi giả lập** vào ChromaDB. Tổng số bản ghi tăng từ `17725` lên `17767` bản ghi.

---

## Ví dụ câu hỏi giả lập sinh bởi AI

### Nhóm Bệnh lý (Mẫu từ Alzheimer)
> [!NOTE]
> - "Dạo này bố mẹ mình cứ quên mất những việc quan trọng và khó tập trung vào một việc là bị bệnh gì ạ?"
> - "Làm sao để biết mình bị bệnh Alzheimer sớm vậy dược sĩ?"
> - "Bị Alzheimer thì có cần kiêng ăn đồ cay nóng không và có cách nào để ngăn chặn sự tiến triển của bệnh không?"

### Nhóm Sản phẩm (Mẫu từ Gel Salonpas)
> [!NOTE]
> - "Gel Salonpas này giá bao nhiêu một hộp 30g ạ?"
> - "Em bị đau lưng và nhức cơ, có thể dùng gel Salonpas được không và bôi mấy lần một ngày?"
> - "Gel Salonpas này có cần bác sĩ kê đơn mới mua được không ạ?"

---

## Hướng dẫn sử dụng cho toàn bộ dữ liệu (600 bệnh & thuốc)

Để chạy sinh câu hỏi và nạp toàn bộ dữ liệu y khoa và sản phẩm vào ChromaDB, ông giáo chạy lệnh dưới đây trong môi trường terminal:

```bash
# Chạy toàn bộ dữ liệu không giới hạn và tự động nạp vào ChromaDB
venv\Scripts\python scripts/s3_generate_synthetic_qa.py --ingest
```

*(Do cơ chế checkpoint đã được lập trình sẵn, script sẽ tự động bỏ qua các bệnh/sản phẩm đã được sinh ở lượt chạy thử nghiệm này và tiếp tục xử lý các phần còn lại).*
