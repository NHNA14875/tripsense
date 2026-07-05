"""
preprocessing.py
=================
Modul preprocessing teks untuk project TripSense (ABSA + NER Hotel Review).
Mencakup kebutuhan Soal 02:
    1. Case folding, tokenization, cleaning URL/mention/simbol, normalisasi slang/typo,
       stopword removal, stemming (Sastrawi jika tersedia).
    2. Regular expression untuk >= 2 fungsi (ekstraksi rating, deteksi kandidat aspek).
    3. Representasi teks: BoW / N-Gram / TF-IDF.
    4. Similarity: minimum edit distance & cosine similarity.

Cara pakai cepat (lihat juga notebooks/eksperimen_absa_ner.ipynb):
    from preprocessing import clean_text, extract_rating, tokenize
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# 0. KAMUS NORMALISASI SLANG/TYPO (Soal 2, poin 1)
#    Ini contoh awal, WAJIB kamu perluas sendiri berdasarkan data asli.
# ---------------------------------------------------------------------------
SLANG_DICT = {
    "bgt": "banget",
    "recomend": "rekomendasi",
    "gk": "tidak",
    "ga": "tidak",
    "nginep": "menginap",
    "kmr": "kamar",
    "kmar": "kamar",
    "kamr": "kamar",
    "yg": "yang",
    "dgn": "dengan",
    "utk": "untuk",
    "sm": "sama",
    "jd": "jadi",
    "krn": "karena",
    "tp": "tapi",
    "bnyk": "banyak",
    "bersi": "bersih",
}

# Stopword sederhana (idealnya pakai Sastrawi StopWordRemoverFactory untuk hasil lebih baik)
STOPWORDS_ID = {
    "yang", "di", "ke", "dari", "dan", "atau", "ini", "itu", "saya", "kami",
    "anda", "untuk", "dengan", "pada", "adalah", "ada", "tidak", "juga",
    "saat", "akan", "sudah", "belum", "karena", "jadi", "tapi", "sangat",
    "sekali", "banget",
}


# ---------------------------------------------------------------------------
# 1. CLEANING & CASE FOLDING
# ---------------------------------------------------------------------------
def case_folding(text: str) -> str:
    """Ubah semua teks ke huruf kecil."""
    return text.lower()


def clean_text(text: str) -> str:
    """
    Bersihkan teks dari:
    - URL
    - mention (@username)
    - hashtag simbol (#)
    - simbol/karakter non-alfanumerik berlebih
    - normalisasi unicode (misal emoji dihilangkan)
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URL
    text = re.sub(r"@\w+", " ", text)                       # mention
    text = re.sub(r"#", " ", text)                           # hashtag symbol
    text = re.sub(r"[^\w\s.,!?]", " ", text)                # simbol aneh/emoji
    text = re.sub(r"\s+", " ", text).strip()                # spasi berlebih
    return text


def normalize_slang(text: str) -> str:
    """Ganti kata slang/singkatan sesuai SLANG_DICT."""
    words = text.split()
    normalized = [SLANG_DICT.get(w, w) for w in words]
    return " ".join(normalized)


def tokenize(text: str) -> list:
    """Tokenisasi sederhana berbasis whitespace + pemisahan tanda baca."""
    text = re.sub(r"([.,!?])", r" \1 ", text)
    return [t for t in text.split() if t.strip()]


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOPWORDS_ID]


def preprocess_pipeline(text: str, use_stopword_removal: bool = True) -> str:
    """
    Pipeline lengkap: case folding -> cleaning -> normalisasi slang -> tokenisasi
    -> (opsional) stopword removal -> gabung kembali jadi string.
    Gunakan fungsi ini untuk menghasilkan tabel before-after preprocessing.
    """
    text = case_folding(text)
    text = clean_text(text)
    text = normalize_slang(text)
    tokens = tokenize(text)
    if use_stopword_removal:
        tokens = remove_stopwords(tokens)
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# 2. REGULAR EXPRESSION (Soal 2, poin 2) — minimal 2 fungsi
# ---------------------------------------------------------------------------
RATING_PATTERN = re.compile(r"(\d(?:[.,]\d)?)\s*(?:/\s*5|dari\s*5|bintang)", re.IGNORECASE)

ASPECT_KEYWORDS = {
    "kebersihan": r"\b(bersih|kotor|debu|bau|apek)\w*\b",
    "pelayanan": r"\b(pelayanan|staf|resepsionis|ramah|jutek|check[- ]?in|check[- ]?out)\w*\b",
    "lokasi": r"\b(lokasi|strategis|akses|dekat|jauh|terpencil)\w*\b",
    "fasilitas": r"\b(fasilitas|wifi|kolam|gym|ac\b)\w*\b",
    "harga": r"\b(harga|mahal|murah|worth)\w*\b",
    "sarapan": r"\b(sarapan|breakfast|menu|buffet)\w*\b",
    "kamar": r"\b(kamar|kasur|sprei|view)\w*\b",
    "parkir": r"\b(parkir|parkiran)\w*\b",
}


def extract_rating(text: str):
    """
    Fungsi regex #1: ekstraksi rating angka dari teks review, misal
    "4/5", "4.5 dari 5", "bintang 3" -> mengembalikan float rating atau None.
    """
    match = RATING_PATTERN.search(text)
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def detect_candidate_aspects(text: str) -> list:
    """
    Fungsi regex #2: deteksi kandidat aspek berdasarkan kata kunci pola regex.
    Berguna sebagai fitur tambahan / validasi sebelum masuk ke model ABSA.
    """
    text_lower = text.lower()
    found = []
    for aspect, pattern in ASPECT_KEYWORDS.items():
        if re.search(pattern, text_lower):
            found.append(aspect)
    return found


# ---------------------------------------------------------------------------
# 3. SIMILARITY (Soal 2, poin 4)
# ---------------------------------------------------------------------------
def edit_distance(s1: str, s2: str) -> int:
    """
    Minimum edit distance (Levenshtein distance) — implementasi manual
    dengan dynamic programming, agar bisa dijelaskan langkah demi langkah
    saat verifikasi lisan.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # hapus
                dp[i][j - 1] + 1,      # sisip
                dp[i - 1][j - 1] + cost,  # ganti
            )
    return dp[m][n]


def correct_typo(word: str, vocabulary: list, max_distance: int = 2):
    """
    Koreksi typo sederhana: cari kata di `vocabulary` dengan edit distance
    terkecil terhadap `word`. Contoh pemakaian untuk aspect normalization,
    misal "kamr" -> "kamar".
    """
    best_word, best_dist = None, max_distance + 1
    for candidate in vocabulary:
        dist = edit_distance(word, candidate)
        if dist < best_dist:
            best_word, best_dist = candidate, dist
    return best_word if best_dist <= max_distance else word


def cosine_similarity_manual(vec1: dict, vec2: dict) -> float:
    """
    Cosine similarity dari dua representasi vektor dalam bentuk dict
    {kata: bobot}. Berguna untuk dijelaskan manual tanpa bergantung sklearn.
    """
    common_words = set(vec1.keys()) & set(vec2.keys())
    dot_product = sum(vec1[w] * vec2[w] for w in common_words)

    norm1 = sum(v ** 2 for v in vec1.values()) ** 0.5
    norm2 = sum(v ** 2 for v in vec2.values()) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def text_to_bow_vector(text: str) -> dict:
    """Bag-of-Words sederhana (hitung frekuensi kata) dari teks yang sudah bersih."""
    tokens = text.split()
    vec = {}
    for t in tokens:
        vec[t] = vec.get(t, 0) + 1
    return vec


# ---------------------------------------------------------------------------
# DEMO CEPAT (jalankan: python src/preprocessing.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    contoh = "Nginep di Hotel Santika, kamr nya bersi bgt tp wifi nya lelet parah! rating 4/5 recomend deh @admin http://x.com"
    print("RAW      :", contoh)
    print("CLEANED  :", preprocess_pipeline(contoh))
    print("RATING   :", extract_rating(contoh))
    print("ASPEK    :", detect_candidate_aspects(contoh))
    print("EDIT DIST (kamr vs kamar):", edit_distance("kamr", "kamar"))
    print("KOREKSI TYPO 'kamr' ->", correct_typo("kamr", ["kamar", "kasur", "kolam"]))

    v1 = text_to_bow_vector("kamar bersih nyaman")
    v2 = text_to_bow_vector("kamar bersih sekali")
    print("COSINE SIMILARITY:", round(cosine_similarity_manual(v1, v2), 3))
