from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_dir = "./distilled-distilbert-3layer"   # your saved folder
device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
model.eval()

text = "I loved the concert last night!"
inputs = tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)

with torch.no_grad():
    logits = model(**inputs).logits
probs = logits.softmax(dim=-1).cpu().numpy()[0]
pred_id = probs.argmax()

# Optionally map id -> label if present in config
id2label = model.config.id2label if hasattr(model.config, "id2label") else None
label = id2label[pred_id] if id2label else str(pred_id)

print("label:", label, "prob:", probs[pred_id])
