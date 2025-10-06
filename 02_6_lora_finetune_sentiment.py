#!/usr/bin/env python3
"""
LoRA/PEFT fine-tuning for sentiment classification on dair-ai/emotion (DistilBERT).
- Updates only a small set of low-rank adapter weights (parameter-efficient).
- Uses Hugging Face PEFT + Trainer.
- Reports accuracy and macro-F1 on validation.
- Saves both the LoRA adapter and an optional merged model.
"""

import os
import numpy as np
import torch
from pathlib import Path
import evaluate

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)

from peft import LoraConfig, get_peft_model, PeftModel

from sklearn.metrics import classification_report

# ---------------------------
# 0) Reproducibility & device
# ---------------------------
SEED = 42
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------------------
# 1) Load dataset
# ---------------------------
emotions = load_dataset("dair-ai/emotion")
label_feature = emotions["train"].features["label"]
label_names = label_feature.names if hasattr(label_feature, "names") else None
num_labels = len(label_names) if label_names else int(label_feature.num_classes)
print("Labels:", label_names, "num_labels:", num_labels)

# ---------------------------
# 2) Tokenizer & tokenization
# ---------------------------
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256)

tokenized = emotions.map(tokenize, batched=True, remove_columns=[c for c in emotions["train"].column_names if c not in ["text","label"]])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---------------------------
# 3) Base model + LoRA config
# ---------------------------
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt, num_labels=num_labels
).to(device)

# Target modules for DistilBERT: attention q_lin / v_lin are common
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"],
    bias="none",
    task_type="SEQ_CLS",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # sanity check

# ---------------------------
# 4) Metrics (accuracy + macro-F1)
# ---------------------------
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    out = {}
    out.update(accuracy.compute(predictions=preds, references=labels))
    out.update(f1.compute(predictions=preds, references=labels, average="macro"))
    return out

# ---------------------------
# 5) TrainingArguments & Trainer
# ---------------------------
outdir = Path("./distilbert-emotion-lora")
args = TrainingArguments(
    output_dir=str(outdir),
    learning_rate=2e-4,                 # LoRA can use a slightly higher LR
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    gradient_accumulation_steps=1,
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
# 6) Train & evaluate
# ---------------------------
trainer.train()
print("\nBest checkpoint:", trainer.state.best_model_checkpoint)
metrics = trainer.evaluate()
print("\nValidation metrics:", metrics)

# Detailed per-class report
preds = trainer.predict(tokenized["validation"])
y_true = preds.label_ids
y_hat = preds.predictions.argmax(axis=1)
print("\nClassification report (validation):")
target_names = label_names if label_names else [str(i) for i in range(num_labels)]
print(classification_report(y_true, y_hat, target_names=target_names, digits=4))

# ---------------------------
# 7) Save LoRA adapter and merged model (optional)
# ---------------------------
save_dir = Path("./distilbert-emotion-lora-adapter")
save_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(save_dir)        # saves adapter weights
tokenizer.save_pretrained(save_dir)

# Optionally merge LoRA into base weights for standalone deployment
merged_dir = Path("./distilbert-emotion-lora-merged")
try:
    base = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device)
    peft_loaded = PeftModel.from_pretrained(base, save_dir)
    merged = peft_loaded.merge_and_unload()
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"\nSaved merged model to: {merged_dir.resolve()}")
except Exception as e:
    print(f"\n[Warning] Could not merge LoRA weights automatically: {e}")
    print("Adapter-only weights are available and can be loaded with PEFT.")

print(f"\nSaved LoRA adapter to: {save_dir.resolve()}")
print("Done.")
