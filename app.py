import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(page_title="Random Forest Classification")

st.title("🧑‍💼 Social Network Ads Prediction")
st.write("Random Forest Classification Model")

# ================= LOAD MODEL =================
if not os.path.exists("rf_classifier.pkl") or not os.path.exists("scaler.pkl"):
    st.error("❌ Model or scaler file not found")
    st.stop()

with open("rf_classifier.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.success("✅ Model loaded successfully")

st.divider()

# ================= INPUTS =================
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, step=1)
salary = st.number_input("Estimated Salary", min_value=0)

gender_value = 1 if gender == "Male" else 0

# ================= PREDICTION =================
if st.button("🔍 Predict"):
    input_data = np.array([[gender_value, age, salary]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ User WILL purchase the product")
    else:
        st.warning("❌ User will NOT purchase the product")
