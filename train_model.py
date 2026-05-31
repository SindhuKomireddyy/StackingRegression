import streamlit as st
import pickle
import numpy as np

# =====================================================
# Load Model and Scaler
# =====================================================

with open("models/stacking_regressor.pkl", "rb") as file:
    model = pickle.load(file)

with open("models/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# =====================================================
# Streamlit UI
# =====================================================

st.set_page_config(
    page_title="Diabetes Progression Prediction",
    page_icon="🩺"
)

st.title("🩺 Diabetes Progression Prediction")
st.write("Enter patient details below")

# =====================================================
# User Inputs
# =====================================================

age = st.number_input("Age", value=0.05)

sex = st.number_input("Sex", value=0.05)

bmi = st.number_input("BMI", value=0.05)

bp = st.number_input("Blood Pressure", value=0.05)

s1 = st.number_input("S1", value=0.05)

s2 = st.number_input("S2", value=0.05)

s4 = st.number_input("S4", value=0.05)

s5 = st.number_input("S5", value=0.05)

s6 = st.number_input("S6", value=0.05)

# =====================================================
# Prediction
# =====================================================

if st.button("Predict"):

    # Feature Engineering

    bmi_age = bmi * age

    bp_s5 = bp * s5

    s1_s2_ratio = s1 / (s2 + 1e-5)

    features = np.array([[
        age,
        sex,
        bmi,
        bp,
        s4,
        s5,
        s6,
        bmi_age,
        bp_s5,
        s1_s2_ratio
    ]])

    # Scaling

    features = scaler.transform(features)

    # Prediction

    prediction = model.predict(features)

    st.success(
        f"Predicted Disease Progression Score: {prediction[0]:.2f}"
    )