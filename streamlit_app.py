import streamlit as st

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction")

st.write("""
Selamat datang pada aplikasi Prediksi Penyakit Jantung menggunakan
algoritma Support Vector Machine (SVM).

Silakan pilih menu di sebelah kiri.

Menu tersedia:

- Training Model
- Prediksi
- Tentang
""")