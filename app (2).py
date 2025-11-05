import streamlit as st
import pandas as pd
import joblib

model = joblib.load("heart_model.joblib")

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered",
)

st.title("❤️ Heart Disease Prediction App")

age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0–3)", [0,1,2,3])
trestbps = st.number_input("Resting BP", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 250)
fbs = st.selectbox("Fasting Blood Sugar >120mg/dl", [1,0])
restecg = st.selectbox("Rest ECG (0–2)", [0,1,2])
thalach = st.number_input("Max Heart Rate", 60, 250, 150)
exang = st.selectbox("Exercise Angina", [1,0])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope", [0,1,2])
ca = st.selectbox("Major Vessels", [0,1,2,3])
thal = st.selectbox("Thal", [0,1,2,3])

input_data = pd.DataFrame([{
    "age": age,"sex": sex,"cp": cp,"trestbps": trestbps,"chol": chol,"fbs": fbs,
    "restecg": restecg,"thalach": thalach,"exang": exang,"oldpeak": oldpeak,
    "slope": slope,"ca": ca,"thal": thal
}])

if st.button("Predict"):
    prob = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]

    st.subheader("✅ Result")
    if pred == 1:
        st.error(f"⚠ High risk\nProbability: {prob:.2f}")
    else:
        st.success(f"✅ Low risk\nProbability: {prob:.2f}")
