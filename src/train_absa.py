"""
train_absa.py
==============
Training model ABSA (Aspect-Based Sentiment Analysis) untuk project TripSense.
Soal 03: Modeling ABSA dan Evaluasi Klasifikasi.

Pipeline:
    1. Load data/absa_dataset.csv (real data, sumber: Cahyaningtyas et al. 2021)
    2. Split train/test (stratified by sentiment)
    3. Representasi teks: TF-IDF
    4. Model: Logistic Regression (baseline utama) + Naive Bayes (pembanding)
    5. Evaluasi: accuracy, precision, recall, F1, confusion matrix
    6. Error analysis: contoh prediksi salah
    7. Simpan model terbaik ke models/absa_model.joblib

NOTE PENTING: Kita membangun dua model terpisah:
    - Model SENTIMENT: memprediksi sentimen (positif/negatif) dari teks review.
    - Model ASPECT   : memprediksi aspek (kamar/lokasi/dst) dari teks review.
Kombinasi keduanya = ABSA (Aspect-Based Sentiment Analysis) end-to-end:
    teks masuk -> prediksi aspek -> prediksi sentimen per aspek.

Cara pakai:
    python src/train_absa.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "absa_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "aspect", "sentiment"])
    print(f"Total data: {len(df)} baris")
    print("Distribusi aspect:\n", df["aspect"].value_counts(), "\n")
    print("Distribusi sentiment:\n", df["sentiment"].value_counts(), "\n")
    return df


def train_and_evaluate(X_train, X_test, y_train, y_test, task_name, vectorizer):
    """
    Latih 2 model (Logistic Regression & Naive Bayes), bandingkan performanya
    dengan F1-score (macro), lalu kembalikan model terbaik + hasil evaluasi.
    """
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "naive_bayes": MultinomialNB(),
    }

    results = {}
    best_name, best_model, best_f1 = None, None, -1

    for name, model in candidates.items():
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()))

        print(f"--- [{task_name}] Model: {name} ---")
        print(f"Accuracy : {acc:.4f}")
        print(f"F1-macro : {f1_macro:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Confusion matrix (label order:", sorted(y_test.unique()), "):")
        print(cm, "\n")

        results[name] = {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "labels": sorted(y_test.unique().tolist()),
        }

        if f1_macro > best_f1:
            best_name, best_model, best_f1 = name, model, f1_macro

    print(f">>> Model terbaik untuk [{task_name}]: {best_name} (F1-macro={best_f1:.4f})\n")
    return best_name, best_model, vectorizer, results


def error_analysis(model, vectorizer, X_test, y_test, n_examples=5):
    Xte = vectorizer.transform(X_test)
    y_pred = model.predict(Xte)
    errors = []
    for text, true_label, pred_label in zip(X_test, y_test, y_pred):
        if true_label != pred_label:
            errors.append({"text": text, "actual": true_label, "predicted": pred_label})
    return errors[:n_examples]


def main():
    df = load_data()

    # ---------------------------------------------------------------
    # MODEL 1: SENTIMENT CLASSIFICATION (positif / negatif)
    # ---------------------------------------------------------------
    X = df["text"]
    y_sent = df["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_sent, test_size=0.2, random_state=42, stratify=y_sent
    )
    sent_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)
    sent_best_name, sent_best_model, sent_vectorizer, sent_results = train_and_evaluate(
        X_train, X_test, y_train, y_test, "SENTIMENT", sent_vectorizer
    )
    sent_errors = error_analysis(sent_best_model, sent_vectorizer, X_test.tolist(), y_test.tolist())

    # ---------------------------------------------------------------
    # MODEL 2: ASPECT CLASSIFICATION (kamar/lokasi/fasilitas/pelayanan/sarapan/harga)
    # ---------------------------------------------------------------
    y_asp = df["aspect"]
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X, y_asp, test_size=0.2, random_state=42, stratify=y_asp
    )
    asp_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)
    asp_best_name, asp_best_model, asp_vectorizer, asp_results = train_and_evaluate(
        X_train2, X_test2, y_train2, y_test2, "ASPECT", asp_vectorizer
    )
    asp_errors = error_analysis(asp_best_model, asp_vectorizer, X_test2.tolist(), y_test2.tolist())

    # ---------------------------------------------------------------
    # SIMPAN MODEL & LAPORAN
    # ---------------------------------------------------------------
    joblib.dump(
        {
            "sentiment_model": sent_best_model,
            "sentiment_vectorizer": sent_vectorizer,
            "sentiment_model_name": sent_best_name,
            "aspect_model": asp_best_model,
            "aspect_vectorizer": asp_vectorizer,
            "aspect_model_name": asp_best_name,
        },
        MODELS_DIR / "absa_model.joblib",
    )
    print(f"[OK] Model tersimpan di {MODELS_DIR / 'absa_model.joblib'}")

    report_summary = {
        "sentiment_task": {
            "best_model": sent_best_name,
            "results": sent_results,
            "error_examples": sent_errors,
        },
        "aspect_task": {
            "best_model": asp_best_name,
            "results": asp_results,
            "error_examples": asp_errors,
        },
    }
    with open(REPORTS_DIR / "absa_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report_summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] Laporan evaluasi tersimpan di {REPORTS_DIR / 'absa_evaluation_report.json'}")

    print("\nContoh error analysis (SENTIMENT):")
    for e in sent_errors:
        print(" -", e)
    print("\nContoh error analysis (ASPECT):")
    for e in asp_errors:
        print(" -", e)


if __name__ == "__main__":
    main()
