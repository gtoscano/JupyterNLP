#!/usr/bin/env python3
"""
DistilBERT layer-drop + Knowledge Distillation (KD) on dair-ai/emotion (FIXED)
- Fixes variable-length collate error by using DataCollatorWithPadding
- Uses a single shared permutation of indices to align teacher & student batches
- Student keeps only N of 6 DistilBERT layers (default: 3)

# default: keep 3 layers, 3 epochs, bs=32, lr=5e-5, alpha=0.5, T=2.0
python distill_distilbert_3layer_kd.py

# keep only 2 layers, train longer
python distill_distilbert_3layer_kd.py --student_layers 2 --epochs 5

# tweak KD weighting and temperature
python distill_distilbert_3layer_kd.py --alpha 0.7 --temperature 3.0
"""

import argparse, numpy as np, torch
from torch.nn import KLDivLoss, CrossEntropyLoss
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    set_seed,
)
import evaluate
from pathlib import Path

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--student_layers", type=int, default=3)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

args = get_args()
set_seed(args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Data
ds = load_dataset("dair-ai/emotion")
label_names = ds["train"].features["label"].names
num_labels = len(label_names)
print("Labels:", label_names)

teacher_ckpt = "bert-base-uncased"
student_ckpt = "distilbert-base-uncased"

tok_teacher = AutoTokenizer.from_pretrained(teacher_ckpt, use_fast=True)
tok_student = AutoTokenizer.from_pretrained(student_ckpt, use_fast=True)

def tokenize_with(tokenizer, max_length):
    def _fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)
    return _fn

ds_t = ds.map(tokenize_with(tok_teacher, args.max_length), batched=True)
ds_s = ds.map(tokenize_with(tok_student, args.max_length), batched=True)

ds_s = ds_s.rename_column("label", "labels")
ds_t = ds_t.rename_column("label", "labels")   # optional for symmetry

cols = ["input_ids", "attention_mask", "labels"]
ds_t.set_format("torch", columns=cols)
ds_s.set_format("torch", columns=cols)

# Models
teacher = AutoModelForSequenceClassification.from_pretrained(
    teacher_ckpt, num_labels=num_labels
).to(device)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False

student = AutoModelForSequenceClassification.from_pretrained(
    student_ckpt, num_labels=num_labels
).to(device)

# Reduce DistilBERT layers (6 -> N)
keep = max(1, min(args.student_layers, 6))
student.distilbert.transformer.layer = torch.nn.ModuleList(
    student.distilbert.transformer.layer[:keep]
)
student.config.n_layers = keep

# Param count
def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable, 100.0 * trainable / total
tot, train, pct = count_params(student)
print(f"Student params -> total: {tot:,} | trainable: {train:,} ({pct:.2f}%) | layers kept: {keep}")

# Collators (pad dynamically per tokenizer)
teacher_collate = DataCollatorWithPadding(tokenizer=tok_teacher, return_tensors="pt")
student_collate = DataCollatorWithPadding(tokenizer=tok_student, return_tensors="pt")

# Shared permutation for aligned batching
N = len(ds_s["train"])
rng = np.random.RandomState(args.seed)
perm = rng.permutation(N)
last = (N // args.batch_size) * args.batch_size  # drop last partial batch
perm = perm[:last]

# Optim & losses
optim = torch.optim.AdamW(student.parameters(), lr=args.lr)
kd_loss_fn = KLDivLoss(reduction="batchmean")
ce_loss_fn = CrossEntropyLoss()
T = args.temperature
alpha = args.alpha

def kd_step(idx_batch):
    # Build padded teacher & student batches for SAME indices
    batch_t = teacher_collate([ds_t["train"][i] for i in idx_batch])
    batch_s = student_collate([ds_s["train"][i] for i in idx_batch])

    with torch.no_grad():
        t_out = teacher(
            input_ids=batch_t["input_ids"].to(device),
            attention_mask=batch_t["attention_mask"].to(device),
        )
        t_logits = t_out.logits

    s_out = student(
        input_ids=batch_s["input_ids"].to(device),
        attention_mask=batch_s["attention_mask"].to(device),
        labels=batch_s["labels"].to(device),
    )
    s_logits = s_out.logits
    hard_ce = s_out.loss

    s_log_probs = torch.log_softmax(s_logits / T, dim=-1)
    t_probs     = torch.softmax(t_logits / T, dim=-1)
    soft_kd = kd_loss_fn(s_log_probs, t_probs) * (T * T)

    return alpha * soft_kd + (1 - alpha) * hard_ce

# Training
for epoch in range(1, args.epochs + 1):
    student.train()
    total_loss = 0.0
    for start in range(0, len(perm), args.batch_size):
        idx = perm[start:start + args.batch_size].tolist()
        loss = kd_step(idx)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item()
    avg_loss = total_loss / (len(perm) // args.batch_size)

    # Eval
    student.eval()
    accm = evaluate.load("accuracy")
    f1m = evaluate.load("f1")
    # iterate in mini-batches for speed
    val_idx = np.arange(len(ds_s["validation"]))
    for start in range(0, len(val_idx), args.batch_size):
        ids = val_idx[start:start + args.batch_size].tolist()
        batch = student_collate([ds_s["validation"][i] for i in ids])
        with torch.no_grad():
            logits = student(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
        preds = logits.argmax(-1).cpu().numpy()
        accm.add_batch(predictions=preds, references=batch["labels"].numpy())
        f1m.add_batch(predictions=preds, references=batch["labels"].numpy())

    print(f"Epoch {epoch} | Train loss: {avg_loss:.4f} | Val acc: {accm.compute()['accuracy']:.4f} | Val macro-F1: {f1m.compute(average='macro')['f1']:.4f}")

# Save compact student
out_dir = Path(f"./distilled-distilbert-{keep}layer")
out_dir.mkdir(parents=True, exist_ok=True)
student.save_pretrained(out_dir)
tok_student.save_pretrained(out_dir)
print(f"Saved compact student to {out_dir.resolve()}")
