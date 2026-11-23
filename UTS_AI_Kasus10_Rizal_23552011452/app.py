"""app.py - Aplikasi Streamlit prediksi keselamatan penumpang Titanic.

Jalankan:  streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import fitur  # dibutuhkan agar pipeline joblib dapat dimuat

BASE = Path(__file__).resolve().parent
PATH_MODEL = BASE / "models" / "titanic_pipeline.joblib"

st.set_page_config(page_title="Prediksi Penumpang Titanic", page_icon="🚢")
st.title("🚢 Prediksi Keselamatan Penumpang Titanic")


@st.cache_resource
def muat_model():
    return joblib.load(PATH_MODEL)


if not PATH_MODEL.exists():
    st.error("Model belum ada. Jalankan `python latih.py` terlebih dahulu.")
    st.stop()

paket = muat_model()
pipe, meta = paket["pipeline"], paket["metadata"]
pilihan = meta["pilihan"]
nama_kelas = {"1st": "Kelas 1", "2nd": "Kelas 2", "3rd": "Kelas 3", "Crew": "Awak kapal"}
nama_sex = {"Male": "Laki-laki", "Female": "Perempuan"}
nama_age = {"Child": "Anak-anak", "Adult": "Dewasa"}

tab_prediksi, tab_metrik = st.tabs(["Prediksi", "Evaluasi model"])

with tab_prediksi:
    st.subheader("Masukkan data penumpang")
    c1, c2, c3 = st.columns(3)
    kelas = c1.selectbox("Kelas", pilihan["class"], format_func=nama_kelas.get)
    sex = c2.selectbox("Jenis kelamin", pilihan["sex"], format_func=nama_sex.get)
    age = c3.selectbox("Usia", pilihan["age"], format_func=nama_age.get)

    if st.button("Prediksi", type="primary"):
        X = pd.DataFrame([{"class": kelas, "sex": sex, "age": age}])
        proba = pipe.predict_proba(X)[0]
        p_selamat = float(proba[list(pipe.classes_).index(1)])
        ambang = meta["metrik_test"]["threshold"]
        if p_selamat >= ambang:
            st.success(f"Prediksi: **SELAMAT** (probabilitas {p_selamat:.1%})")
        else:
            st.error(f"Prediksi: **TIDAK SELAMAT** (probabilitas selamat {p_selamat:.1%})")
        st.write("Probabilitas tiap kelas:")
        st.bar_chart(pd.DataFrame(
            {"Probabilitas": [1 - p_selamat, p_selamat]},
            index=["Tidak selamat", "Selamat"]))
        st.caption("Data historis 1912; hasil menggambarkan pola data, bukan kepastian individu.")

with tab_metrik:
    m = meta["metrik_test"]
    st.subheader(f"Model terpilih: {m['model']}")
    st.caption(f"Dievaluasi sekali pada data test ({m['n_test']} baris), threshold {m['threshold']}")
    k = st.columns(5)
    k[0].metric("Accuracy", f"{m['accuracy']:.3f}")
    k[1].metric("Precision", f"{m['precision']:.3f}")
    k[2].metric("Recall", f"{m['recall']:.3f}")
    k[3].metric("F1", f"{m['f1']:.3f}")
    k[4].metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    st.write(f"Baseline (selalu tebak 'Tidak selamat'): accuracy {m['baseline_mayoritas']:.3f}")
    st.write("Confusion matrix (baris = aktual, kolom = prediksi):")
    st.dataframe(pd.DataFrame(
        m["confusion_matrix"],
        index=["Aktual: Tidak selamat", "Aktual: Selamat"],
        columns=["Prediksi: Tidak selamat", "Prediksi: Selamat"]))
    st.write("Perbandingan model (cross-validation 5-fold pada data train, rata-rata ± std):")
    baris = []
    for nama, r in meta["cv"].items():
        baris.append({"Model": nama, **{mt: f"{v[0]:.3f} ± {v[1]:.3f}" for mt, v in r.items()}})
    st.dataframe(pd.DataFrame(baris), hide_index=True)
