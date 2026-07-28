# Báo cáo tóm tắt mô hình ML Baseline cho hệ thống Pharmacy_RAG

## 1. Tổng quan kiến trúc mô hình

### 1.1 Mục tiêu của Baseline
Mô hình **ML Baseline** trong dự án `Pharmacy_RAG` được xây dựng như một hệ thống đối chứng độc lập, có nhiệm vụ giải quyết bài toán theo quy trình đơn giản nhưng hiệu quả:

**Nhận câu hỏi người dùng → Phân loại bệnh lý → Tra cứu danh mục thuốc phù hợp**

Đây là một baseline theo hướng **truy xuất dựa trên phân loại** chứ chưa sử dụng cơ chế sinh ngôn ngữ tự nhiên hay truy hồi ngữ nghĩa sâu như RAG. Vì vậy, mô hình này có vai trò rất quan trọng trong đồ án:

- Là **mốc chuẩn so sánh** để đánh giá hệ thống RAG sau này.
- Giúp kiểm chứng mức độ cải thiện của RAG về:
  - độ chính xác trả lời,
  - khả năng gợi ý thuốc,
  - thời gian phản hồi,
  - khả năng xử lý câu hỏi thực tế bằng tiếng Việt.

### 1.2 Luồng hoạt động tổng quát

```text
Câu hỏi đầu vào
    ↓
Tiền xử lý tiếng Việt (word_tokenize)
    ↓
TF-IDF Vectorization
    ↓
LinearSVC dự đoán nhãn bệnh lý
    ↓
Tra cứu danh mục thuốc theo nhãn bệnh
    ↓
So khớp từ khóa đã chuẩn hóa không dấu
    ↓
Trả về danh sách thuốc gợi ý
```

### 1.3 Vai trò “vạch đích benchmark”
Trong bối cảnh phát triển hệ thống RAG, baseline đóng vai trò như một **mốc đối chứng thực nghiệm** giúp đánh giá khách quan toàn bộ hệ thống.

- Nếu RAG không vượt được baseline về chất lượng, độ ổn định hoặc tốc độ, thì việc chuyển sang kiến trúc phức tạp hơn chưa chắc đã mang lại giá trị thực tiễn.
- Nếu RAG vượt baseline rõ rệt, khi đó có cơ sở khoa học để khẳng định kiến trúc mới có hiệu quả hơn.

---

## 2. Chi tiết khâu tiền xử lý dữ liệu

### 2.1 Nguồn dữ liệu đầu vào
Dữ liệu huấn luyện được lấy từ file:

- `data/silver/synthetic_medical_qa.json`

Theo cách triển khai trong `src/baseline_ml/data_processor.py`, cấu trúc dữ liệu có dạng ánh xạ theo từng bệnh:

- **key**: mã nhãn hoặc định danh bệnh lý
- **value**: một object chứa:
  - `display_name`: tên hiển thị của bệnh
  - `questions`: danh sách câu hỏi mẫu của người dùng

Quá trình tiền xử lý tạo ra tập dữ liệu huấn luyện gồm:

- **X**: các câu hỏi sau khi tách từ
- **y**: nhãn bệnh lý tương ứng (`category`)

### 2.2 Cách trích xuất dữ liệu trong code
Trong `MedicalDataProcessor.load_samples()`, hệ thống duyệt từng nhóm bệnh trong JSON và tạo ra các mẫu huấn luyện:

- `text`: câu hỏi sau khi qua `word_tokenize`
- `label`: nhãn bệnh
- `display_name`: tên bệnh hiển thị
- `raw_question`: câu hỏi gốc

### 2.3 Kỹ thuật NLP sử dụng `underthesea`
Mô hình dùng thư viện **underthesea** để thực hiện:

```python
word_tokenize(question, format="text")
```

Đây là bước rất quan trọng vì tiếng Việt là ngôn ngữ đơn lập, có đặc điểm:

- từ có thể gồm nhiều âm tiết,
- khoảng trắng không đồng nghĩa với ranh giới từ,
- câu hỏi thực tế thường mang tính khẩu ngữ.

Việc tách từ giúp:

- tăng độ chính xác biểu diễn ngữ nghĩa,
- giảm nhiễu,
- cải thiện hiệu quả phân loại câu hỏi tiếng Việt.

### 2.4 Ý nghĩa với văn phong khẩu ngữ
Các câu hỏi trong dự án mang tính **khẩu ngữ, đời thường**, không theo cú pháp y khoa chuẩn. Ví dụ:

- “Dạo này bố mẹ mình cứ quên mất những việc quan trọng là bị gì và uống thuốc gì?”
- “Mẹ mình bị đau bụng âm ỉ sau khi ăn, ợ chua và nóng rát cổ họng thì là bệnh gì?”

Các biểu đạt kiểu này có tính biến thể cao, nên việc token hóa là nền tảng để mô hình nhận diện được cụm từ khóa quan trọng.

---

## 3. Chi tiết khâu huấn luyện mô hình

### 3.1 Cấu trúc Pipeline trong Scikit-learn
Mô hình được xây dựng bằng `Pipeline` của Scikit-learn, gồm hai thành phần:

1. **TfidfVectorizer**
2. **LinearSVC**

Theo `src/baseline_ml/trainer.py`:

```python
Pipeline(
    steps=[
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000)),
        ("clf", LinearSVC(C=1.0, max_iter=2000)),
    ]
)
```

### 3.2 Vai trò của `TfidfVectorizer`
`TF-IDF` chuyển văn bản thành vector số dựa trên tần suất xuất hiện của từ/cụm từ trong toàn bộ tập dữ liệu.

#### Tham số quan trọng
- `ngram_range=(1, 2)`
  - Giữ cả unigram và bigram.
  - Hữu ích vì nhiều ý nghĩa nằm ở cụm từ ghép như: `đường huyết`, `tiểu đường`, `ho khan`, `đau bụng`.

- `max_features=50000`
  - Giới hạn số đặc trưng tối đa để kiểm soát độ lớn vector và bộ nhớ.

### 3.3 Lý do chọn `LinearSVC`
`LinearSVC` là thuật toán SVM tuyến tính, rất phù hợp cho bài toán phân loại văn bản:

#### Ưu điểm
- Hiệu quả với dữ liệu thưa, đặc trưng chiều cao như TF-IDF.
- Huấn luyện và suy luận nhanh.
- Ổn định cho bài toán multi-class.
- Thường cho chất lượng tốt trong phân loại văn bản.

### 3.4 Thời gian huấn luyện thực tế
Theo kết quả thực thi từ script `main_baseline.py`, thời gian huấn luyện đạt khoảng:

- **~8.57 giây**

Đây là một con số tốt đối với một baseline văn bản, cho thấy pipeline nhẹ và phù hợp làm đối chứng trong nghiên cứu.

---

## 4. Cơ chế suy luận và tìm kiếm thông minh

### 4.1 Dự đoán nhãn bệnh từ câu hỏi đầu vào
Trong `src/baseline_ml/inference.py`, quá trình suy luận diễn ra theo 2 bước:

1. Tiền xử lý câu hỏi bằng `underthesea.word_tokenize()`
2. Dùng mô hình đã train để dự đoán `category`

### 4.2 Cache DataFrame để tối ưu tốc độ
Một điểm tối ưu quan trọng là file:

- `data/silver/products_kb.csv`

được load **duy nhất một lần** trong hàm `__init__` của `MedicalMLPredictor`, thay vì đọc lại ở mỗi request.

#### Ý nghĩa
- Giảm chi phí I/O đáng kể.
- Tránh việc mở file CSV lặp đi lặp lại.
- Giảm latency từ mức vài giây xuống mức rất thấp.

### 4.3 Unidecode Matching để khắc phục lệch chuẩn dấu tiếng Việt
Đây là cải tiến then chốt ở khâu tra cứu thuốc.

#### Vấn đề
Trong dữ liệu bệnh có thể xuất hiện dạng `Ăn không tiêu`, trong khi tên bệnh dự đoán hoặc mô tả thuốc lại có thể là `An Khong Tieu` hoặc chuỗi có dấu khác chuẩn. Nếu so khớp trực tiếp theo chuỗi gốc, hệ thống dễ báo:

- `No matching products found`

#### Giải pháp
Sử dụng `unidecode()` để chuyển mọi văn bản về:

- chữ thường,
- không dấu,
- loại bỏ ký tự nhiễu.

Ví dụ:

| Chuỗi gốc | Sau chuẩn hóa |
|---|---|
| `An Khong Tieu` | `an khong tieu` |
| `ăn không tiêu` | `an khong tieu` |
| `Thuốc trị ăn không tiêu` | `thuoc tri an khong tieu` |

Sau chuẩn hóa, chuỗi của bệnh và chuỗi mô tả sản phẩm có khả năng khớp tốt hơn.

### 4.4 Tách từ khóa bệnh lý thành cụm ngắn
Để tăng khả năng bắt trúng thuốc, mô hình không chỉ dùng tên bệnh nguyên gốc mà còn tạo thêm các biến thể keyword ngắn hơn.

Ví dụ:
- `benhlaophoi` → `lao phoi`, `lao`
- `ankhongtieu` → `an khong tieu`, `kho tieu`

---

## 5. Kết quả thử nghiệm thực tế

### 5.1 Bộ câu hỏi demo
Trong `main_baseline.py`, mô hình được thử trên 3 câu hỏi mẫu:

1. Alzheimer / suy giảm trí nhớ
2. Ăn không tiêu
3. Bệnh lao phổi

### 5.2 Kết quả đầu ra minh họa

| Câu hỏi mẫu | Bệnh dự đoán | Sản phẩm gợi ý | Ghi chú |
|---|---|---|---|
| “Dạo này bố mẹ mình cứ quên mất những việc quan trọng là bị gì và uống thuốc gì?” | Alzheimer | `Aricept Evess` | Bốc đúng nhóm thuốc đặc trị suy giảm trí nhớ |
| “Mẹ mình bị đau bụng âm ỉ sau khi ăn, ợ chua và nóng rát cổ họng thì là bệnh gì?” | Ăn không tiêu | `LiveSpo Clausy` | Match tốt với nhóm tiêu hóa |
| “Em bị ho khan, sốt nhẹ và mệt mỏi kéo dài, nên uống thuốc gì?” | Bệnh lao phổi | `Pyrazinamide` | Gợi ý đúng nhóm thuốc điều trị lao |

### 5.3 Nhận xét khách quan
#### Ưu điểm
- Phân loại bệnh lý nhanh.
- Độ chính xác phân loại cao.
- Kiến trúc đơn giản, dễ giải thích.
- Tra cứu thuốc đã được tối ưu với chuẩn hóa không dấu.

#### Nhược điểm
- Khâu bốc thuốc vẫn phụ thuộc keyword matching.
- Chưa xử lý ngữ nghĩa sâu.
- Khả năng tổng quát hóa còn giới hạn khi gặp câu hỏi ngoài miền dữ liệu huấn luyện.

---

## 6. Đánh giá tổng thể mô hình Baseline

### 6.1 Điểm mạnh
- Tốc độ huấn luyện nhanh.
- Tốc độ suy luận cao.
- Dễ triển khai, dễ giải thích.
- Phù hợp làm mốc benchmark cho RAG.
- Đã được tối ưu tra cứu thuốc bằng chuẩn hóa không dấu.

### 6.2 Hạn chế
- Không xử lý ngữ nghĩa sâu.
- Phụ thuộc lớn vào chất lượng dữ liệu synthetic.
- Matching thuốc vẫn mang tính từ khóa.

### 6.3 Kết luận
Baseline TF-IDF + LinearSVC trong `Pharmacy_RAG` là một nền tảng hợp lý để:

- kiểm chứng tính khả thi của bài toán,
- tạo mốc so sánh định lượng cho RAG,
- đánh giá tác động của các cải tiến truy hồi ngữ nghĩa sau này.

---

## 7. Tài liệu mã nguồn tham chiếu

| Thành phần | File |
|---|---|
| Tiền xử lý dữ liệu | `src/baseline_ml/data_processor.py` |
| Huấn luyện mô hình | `src/baseline_ml/trainer.py` |
| Suy luận và tra cứu thuốc | `src/baseline_ml/inference.py` |
| Script chạy thử nghiệm | `main_baseline.py` |

