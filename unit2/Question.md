# Câu hỏi
## 1. Thứ tự của một quy trình mô hình hóa ngôn ngữ
-> Tokenizer xử lý văn bản và trả về các ID. Mô hình xử lý các ID và đưa ra dự đoán. Dùng tokenizer tiếp để chuyển đổi dự đoán thành văn bản

## 2. Đầu ra tensor của mô hình Transformer có mấy chiều
-> 3. Độ dài chuỗi, batch size, độ dài lớp ấn

## 3. Tokenizer theo từ phụ
-> WordPiece, BPE, Unigram

## 4. Model head(Đầu mô hình) là gì?
-> Là thành phần bổ sung được tạo thành từ một hoặc một vài lớp, để chuyển đổi các dự đoán transformer thành các đầu ra theo yêu cầu cụ thể.

## 5. AutoModel là?
-> Là cái trả về kiến trúc chính xác dựa trên checkpoint

## 6. SoftMax
-> Áp dụng giới hạn dưới và trên, [0-1]

## 7. result?
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
result = tokenizer.tokenize("Hello!")
``
-> danh sách các chuỗi, mỗi chuỗi là 1 token
