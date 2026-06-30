import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from utils.preprocessing import scaling
from utils.training import train_model

st.title("Training Model SVM")

df = pd.read_csv("data/heart_disease.csv")

st.subheader("Dataset")

st.dataframe(df)

st.write("Jumlah Data :", df.shape)

X = df.drop("HeartDisease", axis=1)

y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train, X_test, scaler = scaling(X_train, X_test)

model = train_model(X_train, y_train)

prediksi = model.predict(X_test)

akurasi = accuracy_score(y_test, prediksi)

st.success(f"Akurasi : {akurasi*100:.2f}%")

st.subheader("Confusion Matrix")

st.write(confusion_matrix(y_test, prediksi))

st.subheader("Classification Report")

st.text(classification_report(y_test, prediksi))

joblib.dump(model, "model/svm_heart_disease_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

st.success("Model berhasil disimpan.")