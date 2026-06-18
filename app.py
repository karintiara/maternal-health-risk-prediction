import streamlit as st
import numpy as np
import joblib

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("Maternal Health Risk Prediction")

st.write(
    "Prediksi tingkat risiko kesehatan ibu hamil menggunakan Random Forest"
)
st.write(
    "Accuracy: 86.21%"
)

age = st.number_input(
    "Age",
    min_value=10,
    max_value=70,
    value=25
)

systolic = st.number_input(
    "Systolic BP",
    value=120
)

diastolic = st.number_input(
    "Diastolic BP",
    value=80
)

bs = st.number_input(
    "Blood Sugar",
    value=7.0
)

bodytemp = st.number_input(
    "Body Temperature",
    value=98.0
)

heartrate = st.number_input(
    "Heart Rate",
    value=75
)

if st.button("Predict"):

    data = np.array([
        [
            age,
            systolic,
            diastolic,
            bs,
            bodytemp,
            heartrate
        ]
    ])

    data_scaled = scaler.transform(data)

    prediction = model.predict(
        data_scaled
    )[0]

    mapping = {
        0: "High Risk",
        1: "Low Risk",
        2: "Mid Risk"
    }

    st.success(
        f"Prediction: {mapping[prediction]}"
    )