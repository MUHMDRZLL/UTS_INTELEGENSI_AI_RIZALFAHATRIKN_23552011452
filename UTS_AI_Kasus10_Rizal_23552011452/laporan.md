# Laporan UTS Intelegensi Buatan - Kasus 10 (Titanic)

Nama: Rizal Fahatrik Nugraha | NIM: 23552011452 | Kelas: TIF RP 23D

## 1. Feature engineering
- **Audit data**: 2201 baris, missing value = 0, tidak ada nilai di luar kategori valid. Tipe data semua kolom teks (kategorik). Ada 2177 baris duplikat karena data hanya punya 3 fitur kategorik (maksimal 16 kombinasi). Duplikat **tidak dibuang**: tiap baris adalah satu orang, sehingga duplikat membawa informasi frekuensi.
- **Kolom dipakai**: class, sex, age. Tidak ada kolom ID atau kolom yang baru diketahui setelah kejadian. `survived` adalah target.
- **Fitur baru**: interaksi `class_sex`, `class_age`, `sex_age`, `class_sex_age`, serta `wanita_atau_anak` (aturan evakuasi "women and children first"). Interaksi membantu model linear (Logistic Regression) karena model ini bersifat aditif dan tidak menangkap interaksi sendiri; mis. efek jenis kelamin sangat berbeda antar kelas.
- **Encoding**: One-Hot Encoding (`handle_unknown="ignore"`) untuk semua fitur. Kelas penumpang tidak diberi ordinal karena kategori `Crew` bukan bagian dari urutan 1st-2nd-3rd.
- **Tanpa kebocoran data**: data dibagi 80:20 (stratified) sebelum encoding. Encoder dipelajari hanya dari data train di dalam pipeline.

## 2. Perbandingan model (5-fold CV, data train)
| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.789 | 0.537 | 0.767 |
| Random Forest | 0.788 | 0.541 | 0.767 |
| Gradient Boosting | 0.790 | 0.538 | 0.768 |

Ketiga model hampir identik karena data hanya memiliki sedikit kombinasi unik, sehingga model yang kompleks tidak punya informasi tambahan untuk dipelajari. Dipilih **Logistic Regression** (ROC-AUC tertinggi; jika selisih < 0,005 dipilih yang paling sederhana dan mudah diinterpretasi).

## 3. Hasil pada data test (dievaluasi satu kali)
Threshold = 0.23, ditentukan dari prediksi out-of-fold pada data train (bukan data test), karena kelas "selamat" minoritas.

| Accuracy | Precision | Recall | F1 | ROC-AUC | Baseline mayoritas |
|---|---|---|---|---|---|
| 0.744 | 0.604 | 0.592 | 0.598 | 0.768 | 0.678 |

Confusion matrix: TN=244, FP=55, FN=58, TP=84.

## 4. Interpretasi
- **Accuracy** hanya dibandingkan dengan baseline (selalu menebak "tidak selamat") karena kelas tidak seimbang; selisihnya menunjukkan nilai tambah model.
- **Precision** 0.60: dari semua yang diprediksi selamat, sebesar itu yang benar-benar selamat. **Recall** 0.59: dari semua penyintas nyata, sebesar itu yang berhasil terdeteksi.
- **F1** menyeimbangkan precision dan recall. **ROC-AUC** 0.77 mengukur kemampuan model mengurutkan penumpang dari yang paling mungkin selamat, tanpa bergantung pada threshold.
- Threshold 0,50 menghasilkan recall rendah; menurunkannya menaikkan recall dengan sedikit mengorbankan precision.
- **Keterbatasan**: hanya 3 fitur kategorik, sehingga banyak penumpang dengan profil sama punya nasib berbeda. Batas atas performa model rendah dan itu bukan kesalahan modeling. Model menggambarkan pola historis, bukan kepastian individu.
- **Deploy**: pipeline lengkap + metadata disimpan di `models/titanic_pipeline.joblib`; aplikasi Streamlit (`app.py`) menerima input mentah (class, sex, age).
