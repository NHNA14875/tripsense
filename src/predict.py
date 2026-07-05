"""
predict.py
==========
Modul untuk memuat model ABSA & NER yang sudah dilatih, dan menyediakan
fungsi prediksi siap pakai untuk aplikasi (app/app.py).

Fungsi utama:
    load_models()                  -> load sekali di awal aplikasi
    predict_absa(text, models)     -> {"aspect": ..., "sentiment": ...}
    predict_ner(text, models)      -> list[(token, label)]
    predict_ner_grouped(text, models) -> list of entitas utuh (bukan per-token BIO)
"""
import re
from pathlib import Path

import joblib

from utils import sent2features
from preprocessing import preprocess_pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


def load_models():
    """Load model ABSA (dict berisi 2 sub-model) dan model NER (CRF)."""
    absa_bundle = joblib.load(MODELS_DIR / "absa_model.joblib")
    ner_model = joblib.load(MODELS_DIR / "ner_model.joblib")
    return {
        "sentiment_model": absa_bundle["sentiment_model"],
        "sentiment_vectorizer": absa_bundle["sentiment_vectorizer"],
        "aspect_model": absa_bundle["aspect_model"],
        "aspect_vectorizer": absa_bundle["aspect_vectorizer"],
        "ner_model": ner_model,
    }


def predict_absa(text: str, models: dict) -> dict:
    """
    Prediksi ABSA end-to-end: teks mentah -> preprocessing -> prediksi aspek
    -> prediksi sentimen. Mengembalikan dict {"aspect":..., "sentiment":...}.
    """
    clean = preprocess_pipeline(text, use_stopword_removal=True)
    if not clean.strip():
        clean = text.lower()

    asp_vec = models["aspect_vectorizer"].transform([clean])
    aspect_pred = models["aspect_model"].predict(asp_vec)[0]

    sent_vec = models["sentiment_vectorizer"].transform([clean])
    sentiment_pred = models["sentiment_model"].predict(sent_vec)[0]

    return {"aspect": aspect_pred, "sentiment": sentiment_pred, "clean_text": clean}


def predict_ner(text: str, models: dict):
    """
    Prediksi NER token-level menggunakan CRF.
    Mengembalikan list of (token, label).
    """
    tokens = re.sub(r"([.,!?])", r" \1 ", text).split()
    if not tokens:
        return []
    features = sent2features(tokens)
    labels = models["ner_model"].predict([features])[0]
    return list(zip(tokens, labels))


def group_entities(token_label_pairs):
    """
    Gabungkan token BIO menjadi entitas utuh, misal
    [("Hotel","B-HOTEL"), ("Santika","I-HOTEL")] -> [("Hotel Santika","HOTEL")]
    """
    entities = []
    current_tokens, current_type = [], None

    for token, label in token_label_pairs:
        if label.startswith("B-"):
            if current_tokens:
                entities.append((" ".join(current_tokens), current_type))
            current_tokens = [token]
            current_type = label[2:]
        elif label.startswith("I-") and current_type == label[2:]:
            current_tokens.append(token)
        else:
            if current_tokens:
                entities.append((" ".join(current_tokens), current_type))
                current_tokens, current_type = [], None

    if current_tokens:
        entities.append((" ".join(current_tokens), current_type))
    return entities


if __name__ == "__main__":
    models = load_models()
    contoh = "Saya menginap di Hotel Santika di Semarang, kamarnya bersih dan wifi kencang."
    print("ABSA:", predict_absa(contoh, models))
    ner_result = predict_ner(contoh, models)
    print("NER (token-level):", ner_result)
    print("NER (entitas utuh):", group_entities(ner_result))
