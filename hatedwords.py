import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset
import ast

data_path = "train.csv"
data = pd.read_csv(data_path)
print(data['label'].unique())
print(data['label'].max())

# 
if "text" not in data.columns or "label" not in data.columns:
    raise KeyError("Dataset phải chứa các cột 'text' và 'label'.")

# 
def convert_label(x):
    try:
        if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
            return ast.literal_eval(x)[0] if ast.literal_eval(x) else 0
        return int(x)  # N
    except (ValueError, SyntaxError):
        return 0  # 

data['label'] = data['label'].apply(convert_label)

#
# num_labels = 2
num_labels = data['label'].max
if data['label'].max() >= num_labels:
    raise ValueError(f"Nhãn không hợp lệ: có nhãn lớn hơn hoặc bằng {num_labels}")

X = data['text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = pd.DataFrame({'text': X_train, 'label': y_train})
test_data = pd.DataFrame({'text': X_test, 'label': y_test})

train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # 
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1)
    accuracy = (predictions == labels).float().mean()
    return {"accuracy": accuracy.item()}

if __name__ == "__main__":
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,  # 
        compute_metrics=compute_metrics,
    )

    trainer.train()
    results = trainer.evaluate