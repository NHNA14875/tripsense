"""
build_real_absa_dataset.py
===========================
Mengambil dataset ASLI review hotel Bahasa Indonesia dari:

    Cahyaningtyas, S., Hatta Fudholi, D., & Fathan Hidayatullah, A. (2021).
    Deep Learning for Aspect-Based Sentiment Analysis on Indonesian Hotels
    Reviews. Kinetik: Game Technology, Information System, Computer Network,
    Computing, Electronics, and Control, 6(3).
    Sumber: https://github.com/siwictyas/Deep-Learning-for-Aspect-Based-
            Sentiment-Analysis-on-Indonesian-Hotels-Reviews

Dataset asli berupa file Excel (data_training.xlsx, sheet "Training") berisi
5.703 baris review hotel berlabel aspect + sentiment (real data, bukan buatan).

Script ini:
1. Membaca data_training.xlsx
2. Mengambil kolom clean text, aspect, sentiment
3. Melakukan sampling stratified supaya tiap aspek terwakili seimbang
4. Menyimpan hasil akhir ke data/absa_dataset.csv (menggantikan starter dataset)

Cara pakai:
    python build_real_absa_dataset.py
"""
import csv
import random
from collections import defaultdict

import openpyxl

random.seed(42)

SOURCE_XLSX = "data/source/data_training.xlsx"
OUTPUT_CSV = "data/absa_dataset.csv"

# Kita fokus pada 6 aspek utama yang relevan untuk aplikasi TripSense,
# dan buang label "lainnya" karena terlalu ambigu/noise untuk baseline model.
ASPECT_MAP = {
    "kamar": "kamar",
    "lokasi": "lokasi",
    "hotel": "fasilitas",   # aspek umum ttg hotel/fasilitas fisik
    "pelayanan": "pelayanan",
    "restoran": "sarapan",  # restoran/sarapan hotel
    "harga": "harga",
}

MAX_PER_ASPECT_SENTIMENT = 130  # target agar total sekitar 6 aspek x 2 sentimen x 130 = 1560 baris


def load_real_data(path):
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb["Training"]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    data = rows[1:]

    result = []
    for r in data:
        clean_text, aspect, sentiment = r[6], r[7], r[8]
        if not clean_text or not aspect or not sentiment:
            continue
        if aspect not in ASPECT_MAP:
            continue
        if sentiment not in ("positive", "negative"):
            continue
        mapped_aspect = ASPECT_MAP[aspect]
        mapped_sentiment = "positif" if sentiment == "positive" else "negatif"
        result.append({
            "text": clean_text.strip(),
            "aspect": mapped_aspect,
            "sentiment": mapped_sentiment,
        })
    return result


def stratified_sample(rows, max_per_group=MAX_PER_ASPECT_SENTIMENT):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["aspect"], r["sentiment"])].append(r)

    sampled = []
    for key, items in groups.items():
        random.shuffle(items)
        sampled.extend(items[:max_per_group])
        print(f"  {key[0]:12s} | {key[1]:8s} -> {min(len(items), max_per_group)} baris (tersedia asli: {len(items)})")
    random.shuffle(sampled)
    return sampled


def main():
    print(f"Membaca dataset asli dari: {SOURCE_XLSX}")
    all_rows = load_real_data(SOURCE_XLSX)
    print(f"Total baris valid setelah filter aspek relevan: {len(all_rows)}\n")

    print("Sampling stratified per (aspect, sentiment):")
    final_rows = stratified_sample(all_rows)
    print(f"\nTotal baris final: {len(final_rows)}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "aspect", "sentiment"])
        writer.writeheader()
        writer.writerows(final_rows)
    print(f"[OK] Disimpan ke {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
