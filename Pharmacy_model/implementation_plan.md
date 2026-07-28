# Kế hoạch triển khai Synthetic Data (3 bước)

Tài liệu này trình bày kế hoạch chi tiết để xây dựng pipeline tự động hóa sinh câu hỏi giả lập (Synthetic QA) từ dữ liệu thô bệnh lý (`medical_kb.csv`) và sản phẩm (`products_kb.csv`), gọi API Groq bằng model `llama-3.1-8b-instant`, và nạp kết quả vào ChromaDB mà không làm mất dữ liệu hiện tại.

## User Review Required

> [!IMPORTANT]
> **1. Điều chỉnh cấu trúc Prompt để tránh lỗi Groq API**
> Khi cấu hình API Groq với tham số `response_format={"type": "json_object"}`, API yêu cầu phản hồi của LLM bắt buộc phải là một đối tượng JSON (bắt đầu bằng `{`), không thể là một mảng JSON thô (bắt đầu bằng `[`). 
>
> Để giải quyết vấn đề này mà vẫn giữ nguyên bản chất prompt của bạn, tôi đề xuất điều chỉnh định dạng đầu ra trong Prompt của LLM thành dạng đối tượng JSON có thuộc tính `"questions"`:
> ```json
> {
>   "questions": [
>     "Câu hỏi 1...",
>     "Câu hỏi 2..."
>   ]
> }
> ```
> Điều này đảm bảo hệ thống chạy mượt mà 100% không bị lỗi parse.

> [!WARNING]
> **2. Tối ưu hóa số lượng gọi API cho Sản phẩm (Products)**
> File `products_kb.csv` chứa tới **7.522 sản phẩm**. Nếu gọi LLM cho từng sản phẩm đơn lẻ:
> - Sẽ tốn 7.522 request, mất nhiều giờ chạy do Rate Limit của Groq Free Tier (30 requests/phút).
> - Để giải quyết, tôi sẽ thiết kế script có tính năng **Lưu tiến trình (Checkpoint)**: Lưu kết quả ngay sau mỗi đợt sinh câu hỏi vào file JSON. Nếu bị ngắt quãng giữa chừng, script sẽ tiếp tục chạy từ vị trí dừng lại mà không gọi trùng lặp. Đồng thời, cho phép cấu hình giới hạn số lượng sản phẩm xử lý để chạy thử nghiệm trước.

## Proposed Changes

### Scripts

#### [NEW] [s3_generate_synthetic_qa.py](file:///d:/TLCN/Pharmacy_RAG/scripts/s3_generate_synthetic_qa.py)
Tạo script Python thực hiện trọn vẹn 3 bước:
1. **Bước 1 (Trích xuất):** Đọc dữ liệu từ `data/silver/medical_kb.csv` và `data/silver/products_kb.csv`.
   - Đối với nhóm Bệnh (Medical): Nhóm các bản ghi theo `category` và gộp trường `text` (tối đa 4000 ký tự) để làm ngữ cảnh đầy đủ cho LLM.
   - Đối với nhóm Sản phẩm (Product): Duyệt qua từng sản phẩm (hỗ trợ checkpoint và giới hạn chạy thử).
2. **Bước 2 (Gọi API sinh câu hỏi):**
   - Sử dụng thư viện `groq` kết nối bằng API Key có sẵn trong dự án.
   - Sử dụng model `llama-3.1-8b-instant` với `response_format={"type": "json_object"}`.
   - Tích hợp cơ chế tự động nghỉ (exponential backoff) khi gặp lỗi Rate Limit (HTTP 429).
   - Lưu tiến trình vào `data/silver/synthetic_medical_qa.json` và `data/silver/synthetic_products_qa.json`.
3. **Bước 3 (Nạp vào ChromaDB):**
   - Đọc kết quả từ các file JSON đã sinh.
   - Kết nối tới cơ sở dữ liệu ChromaDB hiện tại (`data/vector_db`, collection `pharmacy_knowledge`) mà không xóa dữ liệu cũ.
   - Định dạng document cho tìm kiếm không dấu bằng `unidecode`: `display_name | câu hỏi | câu hỏi không dấu`.
   - Thiết lập metadata:
     - `type`: `"synthetic_qa"`
     - `sub_type`: `"medical"` hoặc `"product"`
     - `category`: giữ nguyên category gốc (ví dụ: `BENH:alzheimer` hoặc `SP:siro_ho`)
     - Các trường thông tin bổ sung để phục vụ lọc dữ liệu.

## Verification Plan

### Automated Tests
1. **Kiểm tra cú pháp và kết nối:** Chạy thử script với giới hạn nhỏ (ví dụ: sinh câu hỏi cho 3 bệnh và 3 sản phẩm) bằng lệnh:
   ```bash
   venv\Scripts\python scripts/s3_generate_synthetic_qa.py --limit-med 3 --limit-prod 3 --ingest
   ```
2. **Kiểm tra Vector DB sau khi nạp:** Chạy một script truy vấn nhỏ để kiểm tra xem dữ liệu `synthetic_qa` đã có trong ChromaDB chưa và số lượng bản ghi tăng lên bao nhiêu:
   ```bash
   venv\Scripts\python -c "import chromadb; client = chromadb.PersistentClient(path='data/vector_db'); col = client.get_collection('pharmacy_knowledge'); print('Tổng số bản ghi:', col.count()); print('Số bản ghi synthetic_qa:', len(col.get(where={'type': 'synthetic_qa'})['ids']))"
   ```

### Manual Verification
- Xác nhận các file JSON tạm được tạo ra trong `data/silver/` chứa đúng định dạng câu hỏi tiếng Việt khẩu ngữ tự nhiên.
- Đảm bảo cơ chế checkpoint hoạt động tốt bằng cách chạy script lần thứ 2, kiểm tra xem nó có bỏ qua các phần tử đã sinh trước đó hay không.
