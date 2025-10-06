#!/usr/bin/env python3
"""
Baseline sentiment (no transformer)
Strategy: Bag-of-Words (CountVectorizer) + Logistic Regression
Dataset: dair-ai/emotion
"""
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def main():
    ds = load_dataset("dair-ai/emotion")
    X_train = ds["train"]["text"]
    y_train = ds["train"]["label"]
    X_valid = ds["validation"]["text"]
    y_valid = ds["validation"]["label"]

    # Simple Bag-of-Words with basic English tokenization
    vect = CountVectorizer(lowercase=True, max_features=50000, ngram_range=(1,2))
    Xtr = vect.fit_transform(X_train)
    Xva = vect.transform(X_valid)

    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(Xtr, y_train)

    y_pred = clf.predict(Xva)

    acc = accuracy_score(y_valid, y_pred)
    print(f"Validation accuracy (BoW + LogisticRegression): {acc:.4f}\n")
    print(classification_report(y_valid, y_pred, digits=4))

if __name__ == "__main__":
    main()
