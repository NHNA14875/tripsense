# Data Dictionary — TripSense (ABSA + NER Hotel Review)

## Sumber Data ASLI
Dataset ABSA (dan basis teks untuk NER) diambil dari penelitian akademik publik:

> Cahyaningtyas, S., Hatta Fudholi, D., & Fathan Hidayatullah, A. (2021).
> Deep Learning for Aspect-Based Sentiment Analysis on Indonesian Hotels
> Reviews. Kinetik: Game Technology, Information System, Computer Network,
> Computing, Electronics, and Control, 6(3).
> Repo: https://github.com/siwictyas/Deep-Learning-for-Aspect-Based-Sentiment-Analysis-on-Indonesian-Hotels-Reviews

Dataset asli (`data_training.xlsx`, sheet "Training") berisi **5.703 baris review
hotel asli berlabel aspect + sentiment**. Kita melakukan **stratified sampling**
(lihat `build_real_absa_dataset.py`) untuk mengambil 1.560 baris seimbang di
6 aspek utama x 2 sentimen, agar model tidak bias ke kelas mayoritas.

## 1. absa_dataset.csv
Dataset untuk task Aspect-Based Sentiment Analysis (ABSA). **DATA ASLI**, bukan sintetis.

| Kolom     | Tipe   | Deskripsi                                                        |
|-----------|--------|-------------------------------------------------------------------|
| text      | string | Teks review hotel (sudah melalui cleaning/stemming dari sumber asli) |
| aspect    | string | Aspek: kamar, lokasi, fasilitas, pelayanan, sarapan, harga        |
| sentiment | string | Label sentimen: positif / negatif (dataset asli tidak memiliki label netral) |

**Jumlah baris:** 1.560 (hasil stratified sampling dari 5.330 baris asli yang relevan).
**Catatan:** Dataset asli tidak berlabel netral, sehingga kita mendokumentasikan
keterbatasan ini di laporan (sesuai anjuran soal "netral bila memungkinkan").

## 2. ner_bio_dataset.tsv
Dataset untuk task Named Entity Recognition (NER) dengan skema BIO tagging.
**Konstruksi hybrid:** snippet teks aspek diambil dari data ASLI di atas, lalu
dibungkus kalimat konteks yang menyebut nama hotel & kota di Indonesia (entitas
nyata) agar terbentuk struktur kalimat lengkap dengan entitas HOTEL/LOC/ASPECT.
Lihat `src/build_and_train_ner.py` untuk detail konstruksinya.

| Kolom       | Tipe   | Deskripsi                                       |
|-------------|--------|--------------------------------------------------|
| sentence_id | int    | ID kalimat (baris kosong = pemisah antar kalimat) |
| token       | string | Token/kata dalam kalimat                          |
| label       | string | Label BIO: B-HOTEL, I-HOTEL, B-LOC, B-ASPECT, I-ASPECT, O |

**Skema entitas:**
- `HOTEL`  : nama hotel (misal "Hotel Santika")
- `LOC`    : nama lokasi/kota (misal "Semarang")
- `ASPECT` : kata kunci yang mengindikasikan aspek yang dibahas
- `O`      : bukan entitas (outside)

**Jumlah kalimat:** 400.

## Model yang Dilatih (bukan rule-based murni)
- **ABSA:** TF-IDF + Logistic Regression (dibandingkan dengan Naive Bayes),
  2 model terpisah: klasifikasi sentiment dan klasifikasi aspect.
  Hasil: akurasi sentiment ~89%, akurasi aspect ~91% (lihat `reports/absa_evaluation_report.json`).
- **NER:** Conditional Random Field (CRF) sungguhan via `sklearn-crfsuite`,
  fitur berbasis kata (lowercase, suffix, capitalization, context window ±1).
  Hasil: token accuracy 98,5%, F1-macro entitas 87,8% (lihat `reports/ner_evaluation_report.txt`).

## Etika Penggunaan Data
- Data ABSA bersumber dari repositori publik penelitian akademik yang sudah
  dipublikasikan di jurnal (Kinetik UMM), digunakan untuk kepentingan pembelajaran
  sesuai ketentuan soal ("domain data dapat dipilih ... yang etis dan tidak
  melanggar privasi"). Sumber WAJIB dicantumkan di laporan akademik (bagian Dataset).
- Data NER bersifat hybrid: konten aspek asli + entitas hotel/lokasi yang di-generate
  (nama hotel & kota nyata Indonesia, namun kombinasi kalimatnya buatan) karena
  belum ada dataset publik BIO-tagged untuk domain ini.
- Tidak ada data pribadi (nama tamu, kontak, dsb.) yang digunakan.

