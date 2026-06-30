import streamlit as st
import pandas as pd

from utils.prediction import load_model

st.title("Prediksi Penyakit Jantung")

model, scaler = load_model()

age = st.number_input("Age", 20,100,40)

sex = st.selectbox("Sex",[0,1])

cp = st.number_input("Chest Pain Type",0,3)

trestbps = st.number_input("Resting Blood Pressure")

chol = st.number_input("Cholesterol")

fbs = st.selectbox("Fasting Blood Sugar",[0,1])

restecg = st.number_input("Rest ECG",0,2)

thalach = st.number_input("Maximum Heart Rate")

exang = st.selectbox("Exercise Angina",[0,1])

oldpeak = st.number_input("Old Peak")

slope = st.number_input("Slope",0,2)

ca = st.number_input("CA",0,4)

thal = st.number_input("Thal",0,3)

if st.button("Prediksi"):

    data = pd.DataFrame([[

        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal

    ]], columns=[
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal"
    ])

    data = scaler.transform(data)

    hasil = model.predict(data)

    if hasil[0]==1:
        st.error("Terindikasi Penyakit Jantung")
    else:
        st.success("Tidak Terindikasi Penyakit Jantung")