# UTS Intelegensi Buatan - Kasus 10: Penumpang Titanic

Nama: Rizal Fahatrik Nugraha | NIM: 23552011452 | Kelas: TIF RP 23D

Dataset: `dataset/kasus10_titanic.csv` (target `survived`).

## Cara menjalankan (lokal)
```
python -m pip install -r requirements.txt
python latih.py                  # latih, evaluasi, simpan model, tulis laporan.md
python -m streamlit run app.py   # buka aplikasi di browser
```
Jalankan `python latih.py` dulu di komputer Anda agar model dibuat dengan versi scikit-learn Anda sendiri.

## Cara menjalankan (Google Colab)
Unggah `notebook/kasus10_titanic_colab.ipynb` ke Colab, lalu Run all.

## Isi
- `fitur.py`   : fungsi feature engineering
- `latih.py`   : training (CV di data train), evaluasi test sekali, simpan joblib, tulis laporan
- `app.py`     : aplikasi Streamlit
- `models/`    : hasil pipeline (dibuat ulang saat `latih.py` dijalankan)
- `notebook/`  : notebook Colab
- `laporan.md` : laporan singkat
