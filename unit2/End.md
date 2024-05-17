# Tổng kết

API có thể làm đc mọi việc.

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
```

*model_inputs* là đầu vào cho mô hình

Đệm thêm các phần còn thiếu
```python
# Sẽ đệm thêm vào chuỗi sao cho độ dài bằng độ dài tối đa của chuỗi
model_inputs = tokenizer(sequences, padding="longest")

# Sẽ đệm thêm vào chuỗi sao cho độ dài bằng độ dài tối đa của mô hình
# (512 cho BERT hoặc DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")

# Sẽ đệm thêm vào chuỗi sao cho độ dài bằng độ dài tối đa được chỉ định
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)
```

Cắt bớt các chuỗi
```python
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Sẽ cắt bớt chuỗi cho bằng độ dài tối đa của mô hình
# (512 cho BERT hoặc DistilBERT)
model_inputs = tokenizer(sequences, truncation=True)

# Sẽ cắt bớt chuỗi có độ dài dài hơn độ dài tối đa được chỉ định
model_inputs = tokenizer(sequences, max_length=8, truncation=True)
```

Tokenizer có thể xử lý việc chuyển đổi sang các tensor cụ thể, sau đó đem vào mô hình.
Vd: 
```python
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Trả về tensor PyTorch
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

# Trả về tensor TensorFlow
model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")

# Trả về mảng NumPy
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")
```

## Các kiểu token đặc biệt

```python
sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
```
output

```
[101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102]
[1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012]
```
Giải mã để xem:
```python
print(tokenizer.decode(model_inputs["input_ids"]))
print(tokenizer.decode(ids))
```
```
"[CLS] i've been waiting for a huggingface course my whole life. [SEP]"
"i've been waiting for a huggingface course my whole life."
```
Khác nhau do mô hình đã huấn luyện trước nên có sự khác nhau đầu và cuỗi

## TỔNG KẾT

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
```













