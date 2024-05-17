# Xử lý dữ liệu

VD:
```python
import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

# Tương tự như ví dụ trước
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# Đây là phần mới
batch["labels"] = torch.tensor([1, 1])

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()
```
Data chi có 2 câu nên huấn luyện cho vui, cần bộ dữ liệu lớn hơn. 
MRPC gồm 5.801 cặp câu có nhãn cho biết sự liên kết với nhau

# Load Data

Dùng thư viện datasets
```python
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets
```
```
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
```
Lưu mặc định ở *~/.cache/huggingface/datasets* . Trong chương 2 có thể thay đổi nó bằng biến HF_HOME
Xem 1 câu:
```python
raw_train_dataset = raw_datasets["train"]
raw_train_dataset[0]
```
```
{'idx': 0,
 'label': 1,
 'sentence1': 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
 'sentence2': 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .'}
```

Xem label là nhãn gì:
```python
raw_train_dataset.features
```
```
{'sentence1': Value(dtype='string', id=None),
 'sentence2': Value(dtype='string', id=None),
 'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], names_file=None, id=None),
 'idx': Value(dtype='int32', id=None)}
```

# Tiền xử lý

Dùng tokenizer








