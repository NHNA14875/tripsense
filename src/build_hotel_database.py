"""
build_hotel_database.py
=========================
Membangun "database hotel" LOKAL (offline, gratis, tanpa API/kartu kredit)
untuk fitur "Pilih Hotel" di aplikasi.

PENTING — kejujuran soal data ini:
Dataset asli (Cahyaningtyas et al., 2021) TIDAK punya info "review ini dari
hotel mana". Jadi di sini kita mengelompokkan review-review ASLI itu secara
merata ke beberapa nama hotel (yang juga daftar hotelnya kita definisikan
sendiri) supaya user bisa merasakan pengalaman "pilih hotel -> lihat semua
review-nya otomatis dianalisis" TANPA perlu scraping atau API berbayar.

Ini SIMULASI PENGELOMPOKAN, bukan review sungguhan per hotel tsb.
WAJIB dijelaskan secara transparan di laporan (sama seperti disclaimer
untuk dataset NER hybrid).

Output:
    data/hotel_reviews_db.csv  (kolom: hotel, city, text)

Cara pakai:
    python src/build_hotel_database.py
"""
import random
from pathlib import Path

import pandas as pd

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
ABSA_CSV = BASE_DIR / "data" / "absa_dataset.csv"
OUTPUT_CSV = BASE_DIR / "data" / "hotel_reviews_db.csv"

# Pasangan hotel <-> kota (1-ke-1, supaya konsisten tiap kali dijalankan ulang)
HOTEL_CITY_PAIRS = [
    ("Hotel Grand Candi", "Semarang"),
    ("Hotel Santika", "Semarang"),
    ("Hotel Aston", "Yogyakarta"),
    ("Hotel Tentrem", "Yogyakarta"),
    ("Hotel Ciputra", "Jakarta"),
    ("Hotel Novotel", "Bandung"),
    ("Hotel Gumaya", "Semarang"),
    ("Hotel Patra Jasa", "Bali"),
    ("Hotel Horison", "Surabaya"),
    ("Hotel Ibis", "Malang"),
    ("Hotel Mercure", "Solo"),
    ("Hotel Grand Zuri", "Kuta"),
    ("Hotel Aveline", "Ubud"),
    ("Hotel Swiss-Belinn", "Surabaya"),
    ("Hotel Amaris", "Bandung"),
]


def main():
    df = pd.read_csv(ABSA_CSV)
    texts = df["text"].tolist()
    random.shuffle(texts)

    n_hotels = len(HOTEL_CITY_PAIRS)
    chunk_size = len(texts) // n_hotels

    rows = []
    for i, (hotel, city) in enumerate(HOTEL_CITY_PAIRS):
        start = i * chunk_size
        end = start + chunk_size if i < n_hotels - 1 else len(texts)
        for text in texts[start:end]:
            rows.append({"hotel": hotel, "city": city, "text": text})

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_CSV, index=False)

    print(f"[OK] Database hotel lokal dibuat: {OUTPUT_CSV}")
    print(f"Total {len(result_df)} baris, {n_hotels} hotel, rata-rata {chunk_size} review/hotel\n")
    print(result_df.groupby(["hotel", "city"]).size().reset_index(name="jumlah_review"))


if __name__ == "__main__":
    main()
