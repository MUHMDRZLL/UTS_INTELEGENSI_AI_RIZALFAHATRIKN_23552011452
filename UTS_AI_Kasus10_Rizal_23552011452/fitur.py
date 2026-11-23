"""fitur.py - Fungsi feature engineering untuk Kasus 10 (Titanic).

Semua fungsi di sini bersifat deterministik per baris (tidak menghitung statistik
dari data), sehingga aman dipakai sebelum train/test split dan tidak menyebabkan
kebocoran data. Statistik yang dipelajari (encoding) dihitung di dalam pipeline
scikit-learn hanya dari data train.
"""
import pandas as pd

TARGET = "survived"
KOLOM_ASLI = ["class", "sex", "age"]
NILAI_VALID = {
    "class": ["1st", "2nd", "3rd", "Crew"],
    "sex": ["Male", "Female"],
    "age": ["Child", "Adult"],
    "survived": ["No", "Yes"],
}

# Kolom yang masuk model setelah feature engineering (semuanya kategorik)
FITUR_KATEGORIK = [
    "class", "sex", "age",
    "class_sex", "class_age", "sex_age", "class_sex_age",
    "wanita_atau_anak",
]


def muat_data(path):
    """Baca CSV, rapikan nama kolom/nilai, dan buang kolom indeks bawaan."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("unnamed")]  # kolom indeks (jika ada)
    for kol in NILAI_VALID:
        if kol in df.columns:
            df[kol] = df[kol].astype(str).str.strip().str.title()
    # "Crew" dan "1st" tetap benar setelah title-case; pastikan konsisten
    df["class"] = df["class"].replace({"1St": "1st", "2Nd": "2nd", "3Rd": "3rd"})
    return df


def audit_data(df):
    """Audit data: duplikat, missing value, nilai tidak wajar, tipe data."""
    hasil = {
        "jumlah_baris": int(len(df)),
        "tipe_data": df.dtypes.astype(str).to_dict(),
        "missing_per_kolom": df.isna().sum().to_dict(),
        "baris_duplikat": int(df.duplicated().sum()),
        "kombinasi_unik": int(len(df.drop_duplicates())),
        "nilai_tidak_wajar": {},
    }
    for kol, valid in NILAI_VALID.items():
        if kol in df.columns:
            asing = sorted(set(df[kol].dropna().unique()) - set(valid))
            hasil["nilai_tidak_wajar"][kol] = asing
    return hasil


def tambah_fitur(X):
    """Buat fitur baru dari input mentah (class, sex, age).

    Dipakai sebagai langkah pertama pipeline, sehingga aplikasi Streamlit
    cukup mengirim input mentah.
    """
    X = X[KOLOM_ASLI].copy()
    for kol in KOLOM_ASLI:
        X[kol] = X[kol].astype(str)
    # Interaksi 2 arah dan 3 arah
    X["class_sex"] = X["class"] + "_" + X["sex"]
    X["class_age"] = X["class"] + "_" + X["age"]
    X["sex_age"] = X["sex"] + "_" + X["age"]
    X["class_sex_age"] = X["class"] + "_" + X["sex"] + "_" + X["age"]
    # Fitur domain: aturan evakuasi "women and children first"
    X["wanita_atau_anak"] = (
        (X["sex"] == "Female") | (X["age"] == "Child")
    ).map({True: "Ya", False: "Tidak"})
    return X[FITUR_KATEGORIK]


def siapkan_xy(df):
    """Pisahkan fitur mentah dan target biner (Yes=1, No=0)."""
    X = df[KOLOM_ASLI].copy()
    y = (df[TARGET] == "Yes").astype(int)
    return X, y
