#!/usr/bin/env python3
"""
Layer freezing fine-tuning (DistilBERT) for dair-ai/emotion:
- Freeze ALL transformer layers except the LAST TWO (layers 4 and 5).
- Keep classification head trainable.
- Use Hugging Face Trainer and report accuracy + macro-F1.
"""

import numpy as np
import torch
from pathlib import Path
import evaluate

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from sklearn.metrics import classification_report

SEED = 42
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 1) Load dataset
emotions = load_dataset("dair-ai/emotion")
label_feature = emotions["train"].features["label"]
label_names = label_feature.names if hasattr(label_feature, "names") else None
num_labels = len(label_names) if label_names else int(label_feature.num_classes)
print("Labels:", label_names, "num_labels:", num_labels)

# 2) Tokenization
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256)

tokenized = emotions.map(tokenize, batched=True, remove_columns=[c for c in emotions["train"].column_names if c not in ["text","label"]])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 3) Model + Freeze strategy
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device)

# DistilBERT has 6 transformer layers: indices 0..5
# We will freeze everything EXCEPT layers 4 and 5 (last two).
trainable_layer_ids = {4, 5}

# First, freeze entire base model
for p in model.distilbert.parameters():
    p.requires_grad = False

# Then, unfreeze the last two layers
for i, layer in enumerate(model.distilbert.transformer.layer):
    if i in trainable_layer_ids:
        for p in layer.parameters():
            p.requires_grad = True

# Always train the classification head (and pre-classifier for DistilBERT)
if hasattr(model, "pre_classifier"):
    for p in model.pre_classifier.parameters():
        p.requires_grad = True
if hasattr(model, "classifier"):
    for p in model.classifier.parameters():
        p.requires_grad = True

# Utility: count parameters
def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable, 100.0 * trainable / total

total, trainable, pct = count_params(model)
print(f"Parameters -> total: {total:,} | trainable: {trainable:,} ({pct:.2f}%)")
print("Trainable components: last two transformer layers + classifier (and pre-classifier).")

# 4) Metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        **accuracy.compute(predictions=preds, references=labels),
        **f1.compute(predictions=preds, references=labels, average="macro"),
    }

# 5) Training
outdir = Path("./distilbert-emotion-freeze-last2")
args = TrainingArguments(
    output_dir=str(outdir),
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    report_to="none",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

train_result = trainer.train()
print("\nBest checkpoint:", trainer.state.best_model_checkpoint)
metrics = trainer.evaluate()
print("\nValidation metrics:", metrics)

# Detailed per-class report
preds = trainer.predict(tokenized["validation"])
y_true = preds.label_ids
y_hat = preds.predictions.argmax(axis=1)
target_names = label_names if label_names else [str(i) for i in range(num_labels)]
print("\nClassification report (validation):")
print(classification_report(y_true, y_hat, target_names=target_names, digits=4))
