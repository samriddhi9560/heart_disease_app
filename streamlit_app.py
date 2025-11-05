import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("heart_model.joblib")

st.set_page_config(
    page_title="Heart Disease Detector",
    page_icon="❤️",
    layout="centered",
)

# ---------- HEADER ----------
st.markdown(
    """
    <h1 style='text-align:center; color:#d62828;'>
    ❤️ Heart Disease Prediction App
    </h1>
    <p style='text-align:center; font-size:18px;'>
    Provide patient details to predict likelihood of heart disease
    </p>
    """,
    unsafe_allow_html=True
)

# ---------- INPUT FORM ----------
with st.form("input_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 20, 100, 50)
        trestbps = st.number_input("Resting BP", 80, 200, 120)
        chol = st.number_input("Cholesterol", 100, 600, 250)
        thalach = st.number_input("Max Heart Rate", 60, 250, 150)
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)

    with col2:
        sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
        cp = st.selectbox("Chest Pain Type (0–3)", [0,1,2,3])
        fbs = st.selectbox("Fasting Blood Sugar >120mg/dl", [1,0])
        restecg = st.selectbox("Rest ECG (0–2)", [0,1,2])
        exang = st.selectbox("Exercise Angina", [1,0])
        slope = st.selectbox("Slope (0–2)", [0,1,2])
        ca = st.selectbox("Major Vessels (0–3)", [0,1,2,3])
        thal = st.selectbox("Thal (0–3)", [0,1,2,3])

    submitted = st.form_submit_button("🔍 Predict")

# ---------- PREDICTION ----------
if submitted:
    input_data = pd.DataFrame([{
        "age": age,"sex": sex,"cp": cp,"trestbps": trestbps,"chol": chol,"fbs": fbs,
        "restecg": restecg,"thalach": thalach,"exang": exang,"oldpeak": oldpeak,
        "slope": slope,"ca": ca,"thal": thal
    }])

    prob = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]

    st.subheader("✅ Result")

    if pred == 1:
        st.error(f"⚠️ High risk of heart disease\nProbability: {prob:.2f}")
    else:
        st.success(f"✅ Low risk of heart disease\nProbability: {prob:.2f}")

    st.progress(int(prob * 100))

