"""
build_real_ner_dataset.py
==========================
Membangun dataset NER BIO tagging dari TEKS REVIEW ASLI (bukan template
kosong) yang sudah kita punya di data/absa_dataset.csv (sumber: Cahyaningtyas
et al., 2021 — lihat build_real_absa_dataset.py).

Karena dataset publik untuk NER hotel Bahasa Indonesia (BIO-tagged) belum
tersedia, kita membangun BIO dataset dengan cara:
    1. Ambil snippet review ASLI dari absa_dataset.csv (real text, real kosakata)
    2. Bungkus snippet itu dalam kalimat konteks yang menyebut nama HOTEL dan
       LOKASI (kota) nyata di Indonesia -> memberi entitas HOTEL & LOC yang jelas
    3. Tandai kata kunci ASPECT di dalam snippet asli menggunakan kamus kata
       kunci per aspek (berdasarkan kosakata yang benar-benar muncul di data asli)
    4. Simpan ke data/ner_bio_dataset.tsv

Lalu melatih model CRF (Conditional Random Field) sungguhan menggunakan
sklearn-crfsuite pada dataset ini -> models/ner_model.joblib

Cara pakai:
    python src/build_and_train_ner.py
"""
import random
from pathlib import Path

import joblib
import pandas as pd
import sklearn_crfsuite
from sklearn_crfsuite import metrics as crf_metrics
from sklearn.model_selection import train_test_split

from utils import HOTELS, LOCATIONS, sent2features as _sent2features_from_tokens

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
ABSA_CSV = BASE_DIR / "data" / "absa_dataset.csv"
NER_TSV = BASE_DIR / "data" / "ner_bio_dataset.tsv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

CARRIER_TEMPLATES = [
    "Ulasan tamu di {hotel} {loc} : {snippet}",
    "Saya menginap di {hotel} di {loc} , {snippet}",
    "Pengalaman di {hotel} {loc} , {snippet}",
    "{snippet} , ini pendapat saya soal {hotel} di {loc}",
]

# Kamus kata kunci ASPECT, disusun dari kosakata yang benar-benar muncul
# di teks asli (lihat eksplorasi data pada notebook eksperimen).
ASPECT_KEYWORDS = {
    "sarapan": {"sarapan", "menu", "makanan", "sajian", "cafe", "rooftop", "breakfast", "nasgor"},
    "lokasi": {"lokasi", "akses", "dekat", "jauh", "sepi", "strategis", "kemana_mana"},
    "fasilitas": {"fasilitas", "wifi", "lift", "taman", "lobi", "kolam", "playground"},
    "harga": {"harga", "bayar", "gratis", "murah", "mahal", "deposit"},
    "kamar": {"kamar", "ac", "kasur", "sprei", "tv", "mesin_cuci", "hair_dryer", "smooking_room"},
    "pelayanan": {"pelayanan", "layanan", "pegawai", "satpam", "staff", "ramah", "sopan", "customer"},
}


def tag_snippet(snippet_tokens, aspect):
    """Tandai token snippet: kata kunci aspek -> B-ASPECT/I-ASPECT, lainnya -> O."""
    keywords = ASPECT_KEYWORDS.get(aspect, set())
    labels = []
    prev_was_aspect = False
    for tok in snippet_tokens:
        if tok.lower() in keywords:
            labels.append("I-ASPECT" if prev_was_aspect else "B-ASPECT")
            prev_was_aspect = True
        else:
            labels.append("O")
            prev_was_aspect = False
    return labels


def build_ner_dataset(n_samples=400):
    df = pd.read_csv(ABSA_CSV)
    sampled = df.sample(n=min(n_samples, len(df)), random_state=42).reset_index(drop=True)

    sentences = []
    for _, row in sampled.iterrows():
        snippet = str(row["text"])
        aspect = row["aspect"]
        hotel = random.choice(HOTELS)
        loc = random.choice(LOCATIONS)
        template = random.choice(CARRIER_TEMPLATES)
        full_text = template.format(hotel=hotel, loc=loc, snippet=snippet)

        tokens = full_text.split()
        hotel_tokens = hotel.split()
        loc_tokens = [loc]  # lokasi kita anggap 1 token (nama kota)
        snippet_tokens = snippet.split()
        snippet_labels = tag_snippet(snippet_tokens, aspect)
        snippet_label_map = dict(zip(snippet_tokens, snippet_labels))

        labels = []
        i = 0
        while i < len(tokens):
            if tokens[i:i + len(hotel_tokens)] == hotel_tokens:
                labels.append("B-HOTEL")
                labels.extend(["I-HOTEL"] * (len(hotel_tokens) - 1))
                i += len(hotel_tokens)
                continue
            if tokens[i] == loc:
                labels.append("B-LOC")
                i += 1
                continue
            tok_clean = tokens[i].strip(",.:")
            if tok_clean in snippet_label_map:
                labels.append(snippet_label_map[tok_clean])
            else:
                labels.append("O")
            i += 1

        sentences.append(list(zip(tokens, labels)))
    return sentences


def save_ner_tsv(sentences, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("sentence_id\ttoken\tlabel\n")
        for sid, sent in enumerate(sentences):
            for tok, label in sent:
                f.write(f"{sid}\t{tok}\t{label}\n")
            f.write("\n")


# ---------------------------------------------------------------------------
# FEATURE EXTRACTION UNTUK CRF (definisi asli ada di ner_features.py,
# supaya training dan prediksi di aplikasi memakai fitur yang identik)
# ---------------------------------------------------------------------------
def sent2features(sent):
    tokens = [tok for tok, _ in sent]
    return _sent2features_from_tokens(tokens)


def sent2labels(sent):
    return [label for _, label in sent]


def main():
    print("1) Membangun dataset NER BIO dari teks review asli...")
    sentences = build_ner_dataset(n_samples=400)
    save_ner_tsv(sentences, NER_TSV)
    print(f"   [OK] {len(sentences)} kalimat disimpan ke {NER_TSV}")

    print("\n2) Menyiapkan fitur untuk CRF...")
    X = [sent2features(s) for s in sentences]
    y = [sent2labels(s) for s in sentences]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("3) Training CRF (Conditional Random Field)...")
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
    )
    crf.fit(X_train, y_train)

    print("4) Evaluasi model NER...")
    y_pred = crf.predict(X_test)

    labels = list(crf.classes_)
    if "O" in labels:
        labels.remove("O")  # fokus evaluasi ke entitas, bukan token 'O' yang dominan

    flat_report = crf_metrics.flat_classification_report(
        y_test, y_pred, labels=labels, digits=3, zero_division=0
    )
    print(flat_report)

    token_acc = crf_metrics.flat_accuracy_score(y_test, y_pred)
    f1_macro = crf_metrics.flat_f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0)
    print(f"Token accuracy (semua label termasuk O): {token_acc:.4f}")
    print(f"F1-macro (entitas saja): {f1_macro:.4f}")

    # Contoh kesalahan tagging
    print("\nContoh kesalahan tagging (5 pertama):")
    error_count = 0
    for sent, true_labels, pred_labels in zip(sentences[-len(y_test):], y_test, y_pred):
        tokens = [t for t, _ in sent]
        for tok, true_l, pred_l in zip(tokens, true_labels, pred_labels):
            if true_l != pred_l and error_count < 5:
                print(f"  token='{tok}' actual={true_l} predicted={pred_l}")
                error_count += 1

    joblib.dump(crf, MODELS_DIR / "ner_model.joblib")
    print(f"\n[OK] Model CRF tersimpan di {MODELS_DIR / 'ner_model.joblib'}")

    with open(REPORTS_DIR / "ner_evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(flat_report)
        f.write(f"\nToken accuracy: {token_acc:.4f}\n")
        f.write(f"F1-macro (entitas): {f1_macro:.4f}\n")
    print(f"[OK] Laporan evaluasi tersimpan di {REPORTS_DIR / 'ner_evaluation_report.txt'}")


if __name__ == "__main__":
    main()
