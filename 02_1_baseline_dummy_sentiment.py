#!/usr/bin/env python3
"""
Baseline sentiment (no feature extraction, no fine-tuning)
Strategy: DummyClassifier (predicts majority class)
Dataset: dair-ai/emotion
"""
import numpy as np
from datasets import load_dataset
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    ds = load_dataset("dair-ai/emotion")
    X_train = ds["train"]["text"]
    y_train = ds["train"]["label"]
    X_valid = ds["validation"]["text"]
    y_valid = ds["validation"]["label"]

    # Dummy majority-class classifier
    # DummyClassifier provides simple baseline models:
    # - 'most_frequent': always predicts the most common label
    # - 'stratified': random prediction based on label frequencies
    # - 'uniform': completely random predictions
    # - 'constant': always predicts a user-defined label
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit([[0]] * len(X_train), y_train)  # ignore features; provide dummy 2D array

    # Predict using same dummy feature shape
    y_pred = clf.predict([[0]] * len(X_valid))

    acc = accuracy_score(y_valid, y_pred)
    print(f"Validation accuracy (DummyClassifier/majority): {acc:.4f}\n")
    print(classification_report(y_valid, y_pred, digits=4))

if __name__ == "__main__":
    main()
