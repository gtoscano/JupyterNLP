
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
import torch, evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model + tokenizer
model_dir = "./distilled-distilbert-3layer"
model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Load dataset
ds = load_dataset("dair-ai/emotion")

# Tokenize
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256)
dsv = ds["validation"].map(tokenize, batched=True)
cols = ["input_ids", "attention_mask", "label"]
dsv.set_format(type="torch", columns=cols)

# Create DataLoader with collator
from torch.utils.data import DataLoader
collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
valid_loader = DataLoader(dsv, batch_size=32, collate_fn=collator)

# Metrics
acc = evaluate.load("accuracy")
f1 = evaluate.load("f1")

# Evaluate
model.eval()
with torch.no_grad():
    for batch in valid_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds = logits.argmax(-1).cpu().numpy()
        acc.add_batch(predictions=preds, references=batch["labels"].cpu().numpy())
        f1.add_batch(predictions=preds, references=batch["labels"].cpu().numpy())

print("Validation accuracy:", acc.compute()["accuracy"])
print("Validation macro-F1:", f1.compute(average="macro")["f1"])
