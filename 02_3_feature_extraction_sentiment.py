#!/usr/bin/env python3
"""
Feature-extraction sentiment pipeline (compatible with your notebook):
- Uses DistilBERT as a frozen encoder
- Extracts [CLS] hidden-state features
- Trains Logistic Regression on top
"""

import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------
# 0) Reproducibility & device
# ---------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------------------
# 1) Load the same dataset used in the notebook
# ---------------------------
# Split keys: "train", "validation", "test"
emotions = load_dataset("dair-ai/emotion")

# ---------------------------
# 2) Tokenizer & Frozen Encoder (as in the notebook)
# ---------------------------
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt).to(device)
model.eval()
for p in model.parameters():
    p.requires_grad = False

# ---------------------------
# 3) Tokenize function (aligned with notebook style)
# ---------------------------
def tokenize(batch):
    return tokenizer(batch["text"],
                     padding=True,
                     truncation=True)

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

# Keep only columns required by the encoder + label
columns = ["input_ids", "attention_mask", "label"]
emotions_encoded.set_format(type="torch", columns=columns)

# ---------------------------
# 4) Extract hidden states (CLS vector)
# ---------------------------
@torch.no_grad()
def extract_hidden_states(batch):
    inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
    last_hidden_state = model(**inputs).last_hidden_state  # [B, T, H] = [Batch Size (number of sentences), Number of Tokens, 768 for DistilBERT  )
    cls_vec = last_hidden_state[:, 0].cpu().numpy()        # [B, H] (DistilBERT CLS proxy)
    return {"hidden_state": cls_vec}

emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)

# ---------------------------
# 5) Build numpy arrays for scikit-learn
# ---------------------------
X_train = np.array(emotions_hidden["train"]["hidden_state"])
X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["validation"]["label"])

print("Shapes -> X_train:", X_train.shape, "X_valid:", X_valid.shape)

# ---------------------------
# 6) Train a simple classifier on top of frozen features
# ---------------------------
clf = LogisticRegression(max_iter=3000, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_valid)
acc = accuracy_score(y_valid, y_pred)
print(f"\nValidation accuracy: {acc:.4f}\n")
print(classification_report(y_valid, y_pred, digits=4))
