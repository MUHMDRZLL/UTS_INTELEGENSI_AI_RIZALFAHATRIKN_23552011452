"""latih.py - Pelatihan, evaluasi, dan penyimpanan model Titanic (Kasus 10).

Jalankan:  python latih.py
Opsional:  python latih.py --data dataset/kasus10_titanic.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

import fitur

BASE = Path(__file__).resolve().parent
SEED = 42


def buat_pipeline(model):
    """Pipeline lengkap: input mentah -> fitur baru -> one-hot -> model."""
    prep = ColumnTransformer(
        [("kat", OneHotEncoder(handle_unknown="ignore"), fitur.FITUR_KATEGORIK)]
    )
    return Pipeline([
        ("fitur", FunctionTransformer(fitur.tambah_fitur)),
        ("prep", prep),
        ("model", model),
    ])


def main(path_data):
    df = fitur.muat_data(path_data)

    # 1) AUDIT DATA
    audit = fitur.audit_data(df)
    print("=== AUDIT DATA ===")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    df = df.dropna(subset=fitur.KOLOM_ASLI + [fitur.TARGET])
    for kol, valid in fitur.NILAI_VALID.items():
        df = df[df[kol].isin(valid)]
    # Baris duplikat SENGAJA tidak dibuang: setiap baris adalah satu orang,
    # jadi duplikat membawa informasi frekuensi (berapa banyak orang per kombinasi).

    X, y = fitur.siapkan_xy(df)

    # 2) SPLIT DULU (sebelum encoding / statistik apa pun)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)} | "
          f"Proporsi selamat train={y_train.mean():.3f}, test={y_test.mean():.3f}")

    # 3) BANDINGKAN MODEL DENGAN CROSS-VALIDATION DI DATA TRAIN
    kandidat = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    skor = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ringkasan_cv = {}
    print("\n=== CROSS-VALIDATION (5-fold, data train) ===")
    for nama, model in kandidat.items():
        hasil = cross_validate(buat_pipeline(model), X_train, y_train, cv=cv, scoring=skor)
        ringkasan_cv[nama] = {
            m: [round(float(hasil[f"test_{m}"].mean()), 4),
                round(float(hasil[f"test_{m}"].std()), 4)] for m in skor
        }
        r = ringkasan_cv[nama]
        print(f"{nama:20s} acc={r['accuracy'][0]:.3f} prec={r['precision'][0]:.3f} "
              f"rec={r['recall'][0]:.3f} f1={r['f1'][0]:.3f} auc={r['roc_auc'][0]:.3f}")

    # Pilih model dengan ROC-AUC CV tertinggi (tidak bergantung threshold).
    # Jika selisih AUC < 0.005 (praktis seri), pilih model yang lebih sederhana
    # (urutan kandidat: dari yang paling sederhana).
    auc_maks = max(r["roc_auc"][0] for r in ringkasan_cv.values())
    terbaik = next(n for n in kandidat if ringkasan_cv[n]["roc_auc"][0] >= auc_maks - 0.005)
    print(f"\nModel terpilih (ROC-AUC CV, tie -> paling sederhana): {terbaik}")

    # Tentukan threshold dari prediksi out-of-fold di data TRAIN (bukan data test).
    # Kelas 'selamat' minoritas (~32%), jadi threshold 0.5 membuat recall rendah.
    oof = cross_val_predict(buat_pipeline(kandidat[terbaik]), X_train, y_train,
                            cv=cv, method="predict_proba")[:, 1]
    grid = np.round(np.arange(0.20, 0.71, 0.01), 2)
    f1_grid = [f1_score(y_train, (oof >= t).astype(int)) for t in grid]
    threshold = float(grid[int(np.argmax(f1_grid))])
    print(f"Threshold terpilih dari OOF train (F1 maks): {threshold:.2f}")

    # 4) LATIH ULANG DI SELURUH TRAIN, EVALUASI TEST SATU KALI
    pipe = buat_pipeline(kandidat[terbaik]).fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrik = {
        "model": terbaik,
        "threshold": threshold,
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred)), 4),
        "recall": round(float(recall_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "baseline_mayoritas": round(float(1 - y_test.mean()), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    print("\n=== EVALUASI DATA TEST (satu kali) ===")
    print(json.dumps(metrik, indent=2))
    print(classification_report(y_test, pred, target_names=["Tidak selamat", "Selamat"]))

    # 5) SIMPAN PIPELINE + METADATA
    (BASE / "models").mkdir(exist_ok=True)
    paket = {
        "pipeline": pipe,
        "metadata": {
            "fitur_input": fitur.KOLOM_ASLI,
            "pilihan": {k: v for k, v in fitur.NILAI_VALID.items() if k != "survived"},
            "label": {0: "Tidak selamat", 1: "Selamat"},
            "metrik_test": metrik,
            "cv": ringkasan_cv,
            "audit": audit,
        },
    }
    joblib.dump(paket, BASE / "models" / "titanic_pipeline.joblib")
    with open(BASE / "models" / "metrik.json", "w", encoding="utf-8") as f:
        json.dump({"test": metrik, "cv": ringkasan_cv, "audit": audit}, f,
                  indent=2, ensure_ascii=False)
    print("\nModel tersimpan di models/titanic_pipeline.joblib")
    tulis_laporan(audit, ringkasan_cv, metrik, threshold)


def tulis_laporan(audit, cv, m, threshold):
    """Hasilkan laporan.md (angka diambil langsung dari hasil run)."""
    (tn, fp), (fn, tp) = m["confusion_matrix"]
    baris_cv = "\n".join(
        f"| {n} | {r['accuracy'][0]:.3f} | {r['f1'][0]:.3f} | {r['roc_auc'][0]:.3f} |"
        for n, r in cv.items())
    teks = f"""# Laporan UTS Intelegensi Buatan - Kasus 10 (Titanic)

Nama: Rizal Fahatrik Nugraha | NIM: 23552011452 | Kelas: TIF RP 23D

## 1. Feature engineering
- **Audit data**: {audit['jumlah_baris']} baris, missing value = {sum(audit['missing_per_kolom'].values())}, tidak ada nilai di luar kategori valid. Tipe data semua kolom teks (kategorik). Ada {audit['baris_duplikat']} baris duplikat karena data hanya punya 3 fitur kategorik (maksimal 16 kombinasi). Duplikat **tidak dibuang**: tiap baris adalah satu orang, sehingga duplikat membawa informasi frekuensi.
- **Kolom dipakai**: class, sex, age. Tidak ada kolom ID atau kolom yang baru diketahui setelah kejadian. `survived` adalah target.
- **Fitur baru**: interaksi `class_sex`, `class_age`, `sex_age`, `class_sex_age`, serta `wanita_atau_anak` (aturan evakuasi "women and children first"). Interaksi membantu model linear (Logistic Regression) karena model ini bersifat aditif dan tidak menangkap interaksi sendiri; mis. efek jenis kelamin sangat berbeda antar kelas.
- **Encoding**: One-Hot Encoding (`handle_unknown="ignore"`) untuk semua fitur. Kelas penumpang tidak diberi ordinal karena kategori `Crew` bukan bagian dari urutan 1st-2nd-3rd.
- **Tanpa kebocoran data**: data dibagi 80:20 (stratified) sebelum encoding. Encoder dipelajari hanya dari data train di dalam pipeline.

## 2. Perbandingan model (5-fold CV, data train)
| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
{baris_cv}

Ketiga model hampir identik karena data hanya memiliki sedikit kombinasi unik, sehingga model yang kompleks tidak punya informasi tambahan untuk dipelajari. Dipilih **{m['model']}** (ROC-AUC tertinggi; jika selisih < 0,005 dipilih yang paling sederhana dan mudah diinterpretasi).

## 3. Hasil pada data test (dievaluasi satu kali)
Threshold = {threshold:.2f}, ditentukan dari prediksi out-of-fold pada data train (bukan data test), karena kelas "selamat" minoritas.

| Accuracy | Precision | Recall | F1 | ROC-AUC | Baseline mayoritas |
|---|---|---|---|---|---|
| {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['roc_auc']:.3f} | {m['baseline_mayoritas']:.3f} |

Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}.

## 4. Interpretasi
- **Accuracy** hanya dibandingkan dengan baseline (selalu menebak "tidak selamat") karena kelas tidak seimbang; selisihnya menunjukkan nilai tambah model.
- **Precision** {m['precision']:.2f}: dari semua yang diprediksi selamat, sebesar itu yang benar-benar selamat. **Recall** {m['recall']:.2f}: dari semua penyintas nyata, sebesar itu yang berhasil terdeteksi.
- **F1** menyeimbangkan precision dan recall. **ROC-AUC** {m['roc_auc']:.2f} mengukur kemampuan model mengurutkan penumpang dari yang paling mungkin selamat, tanpa bergantung pada threshold.
- Threshold 0,50 menghasilkan recall rendah; menurunkannya menaikkan recall dengan sedikit mengorbankan precision.
- **Keterbatasan**: hanya 3 fitur kategorik, sehingga banyak penumpang dengan profil sama punya nasib berbeda. Batas atas performa model rendah dan itu bukan kesalahan modeling. Model menggambarkan pola historis, bukan kepastian individu.
- **Deploy**: pipeline lengkap + metadata disimpan di `models/titanic_pipeline.joblib`; aplikasi Streamlit (`app.py`) menerima input mentah (class, sex, age).
"""
    (BASE / "laporan.md").write_text(teks, encoding="utf-8")
    print("Laporan tertulis di laporan.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(BASE / "dataset" / "kasus10_titanic.csv"))
    main(ap.parse_args().data)
