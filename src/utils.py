"""
utils.py
========
Fungsi-fungsi bersama yang dipakai baik saat training maupun saat prediksi,
supaya tidak ada duplikasi logika antara build_and_train_ner.py dan predict.py.
"""

HOTELS = ["Hotel Grand Candi", "Hotel Santika", "Hotel Aston", "Hotel Tentrem",
          "Hotel Ciputra", "Hotel Novotel", "Hotel Gumaya", "Hotel Patra Jasa",
          "Hotel Horison", "Hotel Ibis"]
LOCATIONS = ["Semarang", "Yogyakarta", "Bandung", "Surabaya", "Solo", "Bali",
             "Malang", "Jakarta", "Kuta", "Ubud"]

ASPECT_LABELS = ["kamar", "lokasi", "fasilitas", "pelayanan", "sarapan", "harga"]
SENTIMENT_LABELS = ["positif", "negatif"]


# ---------------------------------------------------------------------------
# CRF FEATURE EXTRACTION (dipakai training di build_and_train_ner.py dan
# prediksi di predict.py, agar keduanya 100% konsisten)
# ---------------------------------------------------------------------------
def word2features(sent_tokens, i):
    word = sent_tokens[i]
    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
    }
    if i > 0:
        prev_word = sent_tokens[i - 1]
        features.update({
            "-1:word.lower()": prev_word.lower(),
            "-1:word.istitle()": prev_word.istitle(),
        })
    else:
        features["BOS"] = True

    if i < len(sent_tokens) - 1:
        next_word = sent_tokens[i + 1]
        features.update({
            "+1:word.lower()": next_word.lower(),
            "+1:word.istitle()": next_word.istitle(),
        })
    else:
        features["EOS"] = True
    return features


def sent2features(tokens):
    """Terima list token mentah (string), kembalikan list fitur untuk CRF."""
    return [word2features(tokens, i) for i in range(len(tokens))]


def sent2labels(sent):
    """sent: list of (token, label) tuple -> kembalikan list label saja."""
    return [label for _, label in sent]


def bio_to_entities(tokens, labels):
    """
    Ubah hasil BIO tagging (list token + list label) menjadi list entitas
    yang lebih mudah ditampilkan di UI, misal:
    [{"text": "Hotel Santika", "label": "HOTEL"}, ...]
    """
    entities = []
    current_tokens = []
    current_label = None

    for tok, label in zip(tokens, labels):
        if label.startswith("B-"):
            if current_label is not None:
                entities.append({"text": " ".join(current_tokens), "label": current_label})
            current_label = label[2:]
            current_tokens = [tok]
        elif label.startswith("I-") and current_label == label[2:]:
            current_tokens.append(tok)
        else:
            if current_label is not None:
                entities.append({"text": " ".join(current_tokens), "label": current_label})
            current_label = None
            current_tokens = []

    if current_label is not None:
        entities.append({"text": " ".join(current_tokens), "label": current_label})

    return entities
