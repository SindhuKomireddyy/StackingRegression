import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(
    open("models/stacking_regressor.pkl", "rb")
)

scaler = pickle.load(
    open("models/scaler.pkl", "rb")
)

st.title("Diabetes Progression Prediction")
st.write("Enter patient details")

age = st.number_input("Age")
sex = st.number_input("Sex")
bmi = st.number_input("BMI")
bp = st.number_input("Blood Pressure")
s1 = st.number_input("S1")
s2 = st.number_input("S2")
s3 = st.number_input("S3")
s4 = st.number_input("S4")
s5 = st.number_input("S5")
s6 = st.number_input("S6")

if st.button("Predict"):

    bmi_age = bmi * age
    bp_s5 = bp * s5
    s1_s2_ratio = s1 / (s2 + 1e-5)

    features = np.array([[
        age,
        sex,
        bmi,
        bp,
        s1,
        s2,
        s3,
        s4,
        s5,
        s6,
        bmi_age,
        bp_s5,
        s1_s2_ratio
    ]])

    features = scaler.transform(features)

    prediction = model.predict(features)

    st.success(
        f"Predicted Disease Progression Score: {prediction[0]:.2f}"
    )