"""
app.py
======
Aplikasi Streamlit TripSense — ABSA + NER untuk Review Hotel.
Soal 05: Aplikasi UI/UX Final.

Cara jalankan:
    streamlit run app/app.py
"""
import sys
import json
from pathlib import Path

import pandas as pd
import streamlit as st

# supaya bisa import modul dari folder src/
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from predict import load_models, predict_absa, predict_ner, group_entities  # noqa: E402

st.set_page_config(page_title="TripSense - ABSA & NER Hotel Review", page_icon="🏨", layout="wide")

ENTITY_COLORS = {
    "HOTEL": "#FDE68A",
    "LOC": "#A7F3D0",
    "ASPECT": "#BFDBFE",
}

SENTIMENT_COLORS = {"positif": "#16A34A", "negatif": "#DC2626"}


@st.cache_resource
def get_models():
    return load_models()


def highlight_ner_html(token_label_pairs):
    """Render token-token dengan warna sesuai label entitas (highlight HTML)."""
    html_parts = []
    for token, label in token_label_pairs:
        if label == "O":
            html_parts.append(token)
        else:
            entity_type = label.split("-")[-1]
            color = ENTITY_COLORS.get(entity_type, "#E5E7EB")
            html_parts.append(
                f'<span style="background-color:{color}; padding:2px 4px; '
                f'border-radius:4px; margin:0 1px;" title="{label}">{token}</span>'
            )
    return " ".join(html_parts)


# ---------------------------------------------------------------------------
# SIDEBAR NAVIGASI
# ---------------------------------------------------------------------------
st.sidebar.title("🏨 TripSense")
st.sidebar.caption("ABSA + NER untuk Review Hotel")
page = st.sidebar.radio(
    "Navigasi",
    ["🏠 Beranda / Input Teks", "📂 Upload CSV", "🔎 Hasil NER", "💬 Hasil ABSA",
     "📊 Dashboard Evaluasi"],
)

models = get_models()

# ---------------------------------------------------------------------------
# HALAMAN 1: BERANDA / INPUT TEKS
# ---------------------------------------------------------------------------
if page == "🏠 Beranda / Input Teks":
    st.title("🏨 TripSense")
    st.markdown("Analisis **Aspect-Based Sentiment Analysis (ABSA)** dan "
                "**Named Entity Recognition (NER)** untuk review hotel berbahasa Indonesia.")

    st.subheader("Masukkan Review Hotel")
    default_text = "Saya menginap di Hotel Santika di Semarang, kamarnya bersih dan wifi kencang, tapi harga agak mahal."
    text_input = st.text_area("Tulis atau tempel review di sini:", value=default_text, height=120)

    if st.button("🔍 Analisis Review", type="primary"):
        if not text_input.strip():
            st.warning("Masukkan teks review terlebih dahulu.")
        else:
            st.session_state["last_text"] = text_input
            st.session_state["last_absa"] = predict_absa(text_input, models)
            st.session_state["last_ner"] = predict_ner(text_input, models)

    if "last_text" in st.session_state:
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 💬 Hasil ABSA")
            absa_result = st.session_state["last_absa"]
            sentiment = absa_result["sentiment"]
            color = SENTIMENT_COLORS.get(sentiment, "#374151")
            st.markdown(f"**Aspek terdeteksi:** `{absa_result['aspect']}`")
            st.markdown(
                f"**Sentimen:** <span style='color:{color}; font-weight:bold;'>{sentiment.upper()}</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"Teks setelah preprocessing: _{absa_result['clean_text']}_")

        with col2:
            st.markdown("### 🔎 Hasil NER")
            ner_result = st.session_state["last_ner"]
            st.markdown(highlight_ner_html(ner_result), unsafe_allow_html=True)
            entities = group_entities(ner_result)
            if entities:
                st.markdown("**Entitas terdeteksi:**")
                for ent_text, ent_type in entities:
                    st.markdown(f"- `{ent_type}` : {ent_text}")
            else:
                st.caption("Tidak ada entitas terdeteksi.")

# ---------------------------------------------------------------------------
# HALAMAN 2: UPLOAD CSV
# ---------------------------------------------------------------------------
elif page == "📂 Upload CSV":
    st.title("📂 Upload File CSV")
    st.markdown("Upload file CSV berisi kolom **`text`** (satu review per baris) "
                "untuk dianalisis sekaligus (batch processing).")

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "text" not in df.columns:
            st.error("File CSV wajib memiliki kolom bernama 'text'.")
        else:
            st.success(f"Berhasil memuat {len(df)} baris data.")
            if st.button("🚀 Analisis Semua Baris", type="primary"):
                progress = st.progress(0, text="Memproses...")
                aspects, sentiments = [], []
                for i, text in enumerate(df["text"].astype(str)):
                    result = predict_absa(text, models)
                    aspects.append(result["aspect"])
                    sentiments.append(result["sentiment"])
                    progress.progress((i + 1) / len(df), text=f"Memproses baris {i+1}/{len(df)}")
                progress.empty()

                df["predicted_aspect"] = aspects
                df["predicted_sentiment"] = sentiments
                st.dataframe(df, use_container_width=True)

                st.markdown("### Ringkasan")
                col1, col2 = st.columns(2)
                with col1:
                    st.bar_chart(df["predicted_aspect"].value_counts())
                with col2:
                    st.bar_chart(df["predicted_sentiment"].value_counts())

                csv_result = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Hasil (CSV)", csv_result,
                                    file_name="hasil_analisis_tripsense.csv", mime="text/csv")
    else:
        st.info("Belum ada file yang diupload. Contoh format CSV: kolom `text` berisi review hotel.")
        sample_df = pd.DataFrame({
            "text": [
                "Kamar bersih dan nyaman, tapi wifi lambat.",
                "Sarapan enak dan variatif setiap hari.",
                "Lokasi jauh dari pusat kota, susah cari makan malam.",
            ]
        })
        st.dataframe(sample_df, use_container_width=True)

# ---------------------------------------------------------------------------
# HALAMAN 3: HASIL NER (eksplorasi lebih detail)
# ---------------------------------------------------------------------------
elif page == "🔎 Hasil NER":
    st.title("🔎 Named Entity Recognition (NER)")
    st.markdown("Model CRF akan mengenali entitas: **HOTEL**, **LOC** (lokasi/kota), "
                "dan **ASPECT** (kata kunci aspek yang dibahas).")

    text_input = st.text_area(
        "Masukkan teks:",
        value="Ulasan tamu di Hotel Aston Semarang: ac dan kasur nyaman, sarapan variatif.",
        height=100,
    )
    if st.button("Jalankan NER"):
        ner_result = predict_ner(text_input, models)
        st.markdown("#### Hasil Highlight")
        st.markdown(highlight_ner_html(ner_result), unsafe_allow_html=True)

        st.markdown("#### Detail Token (BIO Tagging)")
        detail_df = pd.DataFrame(ner_result, columns=["token", "label"])
        st.dataframe(detail_df, use_container_width=True)

        entities = group_entities(ner_result)
        st.markdown("#### Entitas Utuh")
        if entities:
            st.dataframe(pd.DataFrame(entities, columns=["entitas", "tipe"]), use_container_width=True)
        else:
            st.caption("Tidak ada entitas terdeteksi.")

    st.divider()
    st.caption("Legenda warna: 🟨 HOTEL   🟩 LOC   🟦 ASPECT")

# ---------------------------------------------------------------------------
# HALAMAN 4: HASIL ABSA (eksplorasi lebih detail)
# ---------------------------------------------------------------------------
elif page == "💬 Hasil ABSA":
    st.title("💬 Aspect-Based Sentiment Analysis (ABSA)")
    st.markdown("Model TF-IDF + Logistic Regression memprediksi **aspek** yang dibahas "
                "dan **sentimen** (positif/negatif) dari teks review.")

    text_input = st.text_area(
        "Masukkan teks review:",
        value="Pelayanan staf ramah sekali, tapi kamar agak sempit.",
        height=100,
    )
    if st.button("Jalankan ABSA"):
        result = predict_absa(text_input, models)
        col1, col2, col3 = st.columns(3)
        col1.metric("Aspek", result["aspect"])
        col2.metric("Sentimen", result["sentiment"])
        col3.metric("Panjang teks (token)", len(result["clean_text"].split()))
        st.caption(f"Teks setelah preprocessing: _{result['clean_text']}_")

# ---------------------------------------------------------------------------
# HALAMAN 5: DASHBOARD EVALUASI
# ---------------------------------------------------------------------------
elif page == "📊 Dashboard Evaluasi":
    st.title("📊 Dashboard Evaluasi Model")
    st.markdown("Ringkasan performa model ABSA dan NER berdasarkan hasil training "
                "(lihat `reports/` untuk detail lengkap).")

    reports_dir = BASE_DIR / "reports"
    absa_report_path = reports_dir / "absa_evaluation_report.json"
    ner_report_path = reports_dir / "ner_evaluation_report.txt"

    tab1, tab2 = st.tabs(["ABSA", "NER"])

    with tab1:
        if absa_report_path.exists():
            with open(absa_report_path, encoding="utf-8") as f:
                absa_report = json.load(f)

            for task_key, task_label in [("sentiment_task", "Sentiment Classification"),
                                          ("aspect_task", "Aspect Classification")]:
                st.subheader(task_label)
                task = absa_report[task_key]
                best_model = task["best_model"]
                best_result = task["results"][best_model]
                col1, col2 = st.columns(2)
                col1.metric("Model Terbaik", best_model)
                col2.metric("Accuracy", f"{best_result['accuracy']*100:.2f}%")

                cr = best_result["classification_report"]
                cr_df = pd.DataFrame(cr).T
                cr_df = cr_df[cr_df.index.map(lambda x: x not in ("accuracy",))]
                st.dataframe(cr_df, use_container_width=True)

                st.markdown("**Contoh Error Analysis:**")
                for err in task["error_examples"]:
                    st.markdown(f"- _{err['text']}_ → aktual: `{err['actual']}`, prediksi: `{err['predicted']}`")
                st.divider()
        else:
            st.warning("Laporan evaluasi ABSA belum ditemukan. Jalankan `python src/train_absa.py` terlebih dahulu.")

    with tab2:
        if ner_report_path.exists():
            with open(ner_report_path, encoding="utf-8") as f:
                st.code(f.read(), language="text")
        else:
            st.warning("Laporan evaluasi NER belum ditemukan. Jalankan `python src/build_and_train_ner.py` terlebih dahulu.")
