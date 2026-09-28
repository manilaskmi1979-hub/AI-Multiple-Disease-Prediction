"""
train_models.py
----------------
Trains a RandomForestClassifier for each of the 5 diseases and saves:
    models/<disease>_model.pkl   -> trained RandomForest
    models/<disease>_scaler.pkl  -> StandardScaler used before prediction
    models/<disease>_columns.pkl -> exact feature order expected by the model

Run:
    python data_generator.py     # (only if you don't already have real CSVs in data/)
    python train_models.py
"""

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DISEASES = {
    "diabetes": {"file": "diabetes.csv", "target": "Outcome"},
    "heart": {"file": "heart.csv", "target": "target"},
    "liver": {"file": "liver.csv", "target": "Dataset"},
    "kidney": {"file": "kidney.csv", "target": "classification"},
    "parkinsons": {"file": "parkinsons.csv", "target": "status"},
}


def train_one(name, file, target):
    path = os.path.join(DATA_DIR, file)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run data_generator.py first, "
            f"or place the real dataset there with this exact filename."
        )

    df = pd.read_csv(path)
    df = df.dropna()

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_split=2,
        random_state=42, n_jobs=-1
    )
    clf.fit(X_train_s, y_train)

    preds = clf.predict(X_test_s)
    acc = accuracy_score(y_test, preds)

    print(f"\n=== {name.upper()} ===")
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, preds))

    with open(os.path.join(MODEL_DIR, f"{name}_model.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(MODEL_DIR, f"{name}_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, f"{name}_columns.pkl"), "wb") as f:
        pickle.dump(list(X.columns), f)

    return acc


if __name__ == "__main__":
    results = {}
    for name, info in DISEASES.items():
        results[name] = train_one(name, info["file"], info["target"])

    print("\n================ SUMMARY ================")
    for name, acc in results.items():
        print(f"{name.capitalize():12s}: {acc*100:.2f}% accuracy")
    print("All models saved in the models/ folder.")
