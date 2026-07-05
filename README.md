# TripSense — ABSA + NER Hotel Review (Project UAS PBA Berbasis Teks)

Aplikasi analisis review hotel yang mengekstrak aspek + sentimen (ABSA) dan
mengenali entitas (NER: HOTEL, LOC, ASPECT) dari teks review.

## Status Progress
- [x] Struktur folder project
- [x] Dataset ABSA **ASLI** (1.560 baris, sumber: penelitian akademik Cahyaningtyas et al. 2021)
- [x] Dataset NER BIO (400 kalimat, hybrid: teks asli + entitas hotel/lokasi)
- [x] Modul preprocessing (`src/preprocessing.py`)
- [x] Model ABSA **terlatih** (TF-IDF + Logistic Regression, akurasi sentiment ~89%, aspect ~91%)
- [x] Model NER **terlatih** (CRF via sklearn-crfsuite, token accuracy 98,5%)
- [x] Modul prediksi (`src/predict.py`, `src/utils.py`)
- [x] Aplikasi Streamlit (`app/app.py`) — sudah ditest jalan tanpa error
- [ ] Notebook eksperimen konsolidasi (Soal 2 before-after preprocessing)
- [ ] Laporan akademik
- [ ] Presentasi & video demo

## Struktur Folder
```
data/
  source/          -> data_training.xlsx (sumber data asli, opsional dijalankan ulang)
  absa_dataset.csv, ner_bio_dataset.tsv, data_dictionary.md
notebooks/         -> eksperimen_absa_ner.ipynb (menyusul)
src/
  preprocessing.py       -> cleaning, regex, similarity (Soal 2)
  train_absa.py          -> training model ABSA (Soal 3)
  build_and_train_ner.py -> bangun dataset + training CRF NER (Soal 4)
  predict.py             -> fungsi prediksi untuk aplikasi
  utils.py               -> fungsi bersama (feature extraction CRF, dll)
app/
  app.py             -> APLIKASI STREAMLIT (Soal 5) — SUDAH JADI
models/              -> absa_model.joblib, ner_model.joblib
reports/             -> absa_evaluation_report.json, ner_evaluation_report.txt
presentation/        -> slide presentasi (menyusul)
demo/                -> link video demo (menyusul)
proof/               -> screenshot UI (menyusul)
build_real_absa_dataset.py  -> script pengambilan & sampling data asli dari sumber
```

## Fitur "Pilih Hotel" (100% Gratis, Offline, Tanpa API/Kartu Kredit)

Alih-alih scraping (melanggar ToS platform review) atau API berbayar (Google
Places API butuh kartu kredit), fitur "Pilih Hotel" memakai **database hotel
lokal** yang dibangun dari data ABSA asli yang sama, dikelompokkan ke 15 nama
hotel contoh (lihat `src/build_hotel_database.py`).

**Cara membangun ulang database ini (opsional, sudah ada di repo):**
```bash
python src/build_hotel_database.py
```
Ini akan membuat `data/hotel_reviews_db.csv` berisi ~104 review per hotel.

**PENTING untuk laporan/verifikasi lisan:** ini adalah **simulasi pengelompokan**
review ke nama hotel, BUKAN review sungguhan per hotel tersebut (dataset sumber
tidak menyertakan info hotel). Isi teks review tetap asli dari data akademik;
yang disimulasikan hanya "review ini milik hotel yang mana". Jelaskan ini
secara transparan — ini termasuk keputusan teknis yang harus didokumentasikan
sesuai ketentuan soal.

## Cara Menjalankan
```bash
pip install -r requirements.txt

# 1. (opsional, sudah dijalankan) bangun ulang dataset ABSA dari sumber asli
python build_real_absa_dataset.py

# 2. Training model ABSA (sentiment + aspect classification)
python src/train_absa.py

# 3. Bangun dataset NER + training CRF
python src/build_and_train_ner.py

# 4. Jalankan aplikasi
streamlit run app/app.py
```

## Halaman Aplikasi Streamlit
1. **Beranda / Input Teks** — analisis satu review, lihat hasil ABSA + NER + highlight entitas
2. **Pilih Hotel (Database Lokal)** — pilih hotel dari daftar, semua review-nya otomatis dianalisis (gratis, offline)
3. **Upload CSV** — analisis banyak review sekaligus, download hasil sebagai CSV
4. **Hasil NER** — eksplorasi detail token-level BIO tagging
5. **Hasil ABSA** — eksplorasi detail klasifikasi aspek + sentimen
6. **Dashboard Evaluasi** — metrik model, confusion matrix, error analysis
7. **Panduan Penggunaan** — cara pakai + keterbatasan sistem + sumber data

## Hasil Training Saat Ini
| Model | Task | Metode | Akurasi | F1-macro |
|---|---|---|---|---|
| ABSA - Sentiment | Klasifikasi sentimen (positif/negatif) | TF-IDF + Logistic Regression | 89.1% | 89.1% |
| ABSA - Aspect | Klasifikasi aspek (6 kelas) | TF-IDF + Logistic Regression | 91.3% | 91.4% |
| NER | Sequence labeling BIO (HOTEL/LOC/ASPECT) | CRF (sklearn-crfsuite) | 98.5% (token) | 87.8% (entitas) |

## Catatan Penting untuk Hanif
1. **Dataset ABSA sudah dari sumber ASLI** (penelitian akademik terpublikasi),
   bukan buatan sendiri — tapi kamu WAJIB mencantumkan sitasi sumbernya di laporan
   (lihat `data/data_dictionary.md`) supaya tidak dianggap plagiarisme data.
2. **Dataset NER bersifat hybrid**: konten aspek dari data asli, tapi kalimat
   pembungkus (nama hotel/kota dalam satu kalimat) di-generate. Jelaskan ini
   secara transparan di laporan — ini termasuk "keputusan teknis yang harus
   dijelaskan" sesuai ketentuan soal.
3. Pelajari `src/train_absa.py` dan `src/build_and_train_ner.py` baik-baik —
   dosen bisa minta kamu jelaskan kenapa pakai Logistic Regression vs Naive Bayes,
   kenapa CRF, dan bagaimana confusion matrix-nya terbentuk.
4. Langkah selanjutnya: notebook eksperimen (before-after preprocessing) untuk
   Soal 2, lalu integrasi model ke Streamlit untuk Soal 5.
