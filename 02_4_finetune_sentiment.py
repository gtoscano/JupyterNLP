#!/usr/bin/env python3
"""
Fine-tune DistilBERT for sentiment (dair-ai/emotion), aligned with your notebook:

- Uses the same dataset ("dair-ai/emotion")
- Tokenizes with AutoTokenizer
- Fine-tunes AutoModelForSequenceClassification via HF Trainer
- Reports accuracy & macro-F1 on the validation split
- Saves best model and tokenizer; also creates a zip for easy download
"""

import os
import numpy as np
import torch
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from pathlib import Path
import shutil

# ---------------------------
# 0) Reproducibility & device
# ---------------------------
SEED = 42
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------------------
# 1) Load dataset (same as notebook)
# ---------------------------
emotions = load_dataset("dair-ai/emotion")
label_feature = emotions["train"].features["label"]
label_names = label_feature.names if hasattr(label_feature, "names") else None
num_labels = len(label_names) if label_names else int(label_feature.num_classes)
print("Labels:", label_names, "num_labels:", num_labels)

# ---------------------------
# 2) Tokenizer & model
# ---------------------------
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(batch):
    return tokenizer(batch["text"],
                     padding=True,
                     truncation=True)

tokenized = emotions.map(tokenize, batched=True, remove_columns=[c for c in emotions["train"].column_names if c not in ["text","label"]])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt,
    num_labels=num_labels
).to(device)

# ---------------------------
# 3) Metrics (accuracy + macro-F1)
# ---------------------------
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    results = {}
    results.update(accuracy.compute(predictions=preds, references=labels))
    results.update(f1.compute(predictions=preds, references=labels, average="macro"))
    return results

# ---------------------------
# 4) TrainingArguments & Trainer
# ---------------------------
outdir = Path("./distilbert-emotion-ft")
args = TrainingArguments(
    output_dir=str(outdir),
    learning_rate=2e-5,
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

# ---------------------------
# 5) Train & evaluate
# ---------------------------
train_result = trainer.train()
print("\nBest checkpoint:", trainer.state.best_model_checkpoint)
eval_metrics = trainer.evaluate()
print("\nValidation metrics:", eval_metrics)

# ---------------------------
# 6) Save best model & tokenizer, and make a zip for download
# ---------------------------
best_dir = Path(trainer.state.best_model_checkpoint or outdir)
save_dir = Path("./distilbert-emotion-ft-best")
if best_dir.exists():
    # load the best model into memory and save a clean copy
    best_model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    best_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
else:
    # fallback if best checkpoint not found
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

zip_target = Path("./data/distilbert_emotion_ft_best.zip")
if zip_target.exists():
    zip_target.unlink()

shutil.make_archive("./data/distilbert_emotion_ft_best", "zip", root_dir=save_dir)

# ---------------------------
# 7) Optional: detailed classification report on validation
# ---------------------------
preds = trainer.predict(tokenized["validation"])
y_true = preds.label_ids
y_hat = preds.predictions.argmax(axis=1)

print("\nClassification report (validation):")
target_names = label_names if label_names else [str(i) for i in range(num_labels)]
print(classification_report(y_true, y_hat, target_names=target_names, digits=4))

print("\nSaved best model directory:", save_dir.resolve())
print("Zipped model:", zip_target.resolve())
