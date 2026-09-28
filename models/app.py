"""
app.py
------
AI-Based Multiple Disease Prediction — Streamlit Web App
Predicts: Diabetes, Heart Disease, Liver Disease, Kidney Disease, Parkinson's Disease

Run with:
    streamlit run app.py
"""

import os
import pickle
import numpy as np
import streamlit as st

BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "models")

st.set_page_config(page_title="AI Multiple Disease Prediction", page_icon="🩺", layout="wide")


@st.cache_resource
def load_artifacts(name):
    with open(os.path.join(MODEL_DIR, f"{name}_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, f"{name}_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, f"{name}_columns.pkl"), "rb") as f:
        columns = pickle.load(f)
    return model, scaler, columns


def predict(name, values):
    model, scaler, columns = load_artifacts(name)
    X = np.array(values).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][int(pred)]
    return pred, proba


# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("🩺 Disease Prediction")
page = st.sidebar.radio(
    "Choose a disease to check:",
    ["🏠 Home", "Diabetes", "Heart Disease", "Liver Disease", "Kidney Disease", "Parkinson's Disease"]
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ This tool gives a quick ML-based risk estimate only. "
    "It is NOT a medical diagnosis. Please consult a doctor for confirmation."
)

# ---------------- HOME ----------------
if page == "🏠 Home":
    st.title("AI-Based Multiple Disease Prediction")
    st.subheader("Predicting Health, Protecting Lives")
    st.write("""
    This system uses **Machine Learning (Random Forest Classification)** to predict
    the risk of 5 major diseases based on the user's health details:

    - 🍬 **Diabetes**
    - ❤️ **Heart Disease**
    - 🫁 **Liver Disease**
    - 🫘 **Kidney Disease**
    - 🧠 **Parkinson's Disease**

    Select a disease from the left sidebar, enter the health parameters, and get an
    instant prediction with a suggestion.
    """)
    st.info("Tip: Run `python data_generator.py` then `python train_models.py` once before "
            "launching this app, so the models/ folder has the trained .pkl files.")

# ---------------- DIABETES ----------------
elif page == "Diabetes":
    st.title("🍬 Diabetes Prediction")
    c1, c2 = st.columns(2)
    with c1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose Level", 0.0, 300.0, 110.0)
        bp = st.number_input("Blood Pressure", 0.0, 200.0, 72.0)
        skin = st.number_input("Skin Thickness", 0.0, 100.0, 20.0)
    with c2:
        insulin = st.number_input("Insulin", 0.0, 900.0, 80.0)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age = st.number_input("Age", 1, 120, 30)

    if st.button("Predict Diabetes"):
        pred, proba = predict("diabetes", [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age])
        if pred == 1:
            st.error(f"⚠️ High risk of Diabetes (confidence: {proba*100:.1f}%). "
                     "Please consult a doctor and get an HbA1c / fasting glucose test.")
        else:
            st.success(f"✅ Low risk of Diabetes (confidence: {proba*100:.1f}%). "
                       "Maintain a healthy diet and regular exercise.")

# ---------------- HEART ----------------
elif page == "Heart Disease":
    st.title("❤️ Heart Disease Prediction")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 1, 120, 45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", 80.0, 220.0, 130.0)
        chol = st.number_input("Cholesterol", 100.0, 500.0, 240.0)
    with c2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate Achieved", 60.0, 220.0, 150.0)
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    with c3:
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 7.0, 1.0)
        slope = st.selectbox("Slope of ST Segment (0-2)", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia (0-2)", [0, 1, 2])

    if st.button("Predict Heart Disease"):
        values = [
            age, 1 if sex == "Male" else 0, cp, trestbps, chol,
            1 if fbs == "Yes" else 0, restecg, thalach, 1 if exang == "Yes" else 0,
            oldpeak, slope, ca, thal
        ]
        pred, proba = predict("heart", values)
        if pred == 1:
            st.error(f"⚠️ High risk of Heart Disease (confidence: {proba*100:.1f}%). "
                     "Please consult a cardiologist soon.")
        else:
            st.success(f"✅ Low risk of Heart Disease (confidence: {proba*100:.1f}%). "
                       "Keep up a heart-healthy lifestyle.")

# ---------------- LIVER ----------------
elif page == "Liver Disease":
    st.title("🫁 Liver Disease Prediction")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 1, 120, 40)
        gender = st.selectbox("Gender", ["Male", "Female"])
        tb = st.number_input("Total Bilirubin", 0.0, 30.0, 1.0)
        db = st.number_input("Direct Bilirubin", 0.0, 15.0, 0.3)
        alkphos = st.number_input("Alkaline Phosphotase", 0.0, 2000.0, 210.0)
    with c2:
        sgpt = st.number_input("Alamine Aminotransferase (SGPT)", 0.0, 1000.0, 30.0)
        sgot = st.number_input("Aspartate Aminotransferase (SGOT)", 0.0, 1000.0, 35.0)
        tp = st.number_input("Total Proteins", 0.0, 12.0, 6.5)
        alb = st.number_input("Albumin", 0.0, 7.0, 3.2)
        agr = st.number_input("Albumin and Globulin Ratio", 0.0, 4.0, 1.0)

    if st.button("Predict Liver Disease"):
        values = [age, 1 if gender == "Male" else 0, tb, db, alkphos, sgpt, sgot, tp, alb, agr]
        pred, proba = predict("liver", values)
        if pred == 1:
            st.error(f"⚠️ High risk of Liver Disease (confidence: {proba*100:.1f}%). "
                     "Please consult a hepatologist / gastroenterologist.")
        else:
            st.success(f"✅ Low risk of Liver Disease (confidence: {proba*100:.1f}%). "
                       "Avoid excess alcohol and maintain a balanced diet.")

# ---------------- KIDNEY ----------------
elif page == "Kidney Disease":
    st.title("🫘 Kidney Disease Prediction")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 1, 120, 45)
        bp = st.number_input("Blood Pressure", 40.0, 200.0, 80.0)
        sg = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
        al = st.selectbox("Albumin (0-4)", [0, 1, 2, 3, 4])
        su = st.selectbox("Sugar (0-4)", [0, 1, 2, 3, 4])
        bgr = st.number_input("Blood Glucose Random", 50.0, 500.0, 140.0)
    with c2:
        bu = st.number_input("Blood Urea", 1.0, 300.0, 40.0)
        sc = st.number_input("Serum Creatinine", 0.1, 20.0, 1.0)
        sod = st.number_input("Sodium", 100.0, 170.0, 137.0)
        pot = st.number_input("Potassium", 2.0, 12.0, 4.5)
        hemo = st.number_input("Hemoglobin", 3.0, 18.0, 13.0)
        htn = st.selectbox("Hypertension", ["No", "Yes"])
        dm = st.selectbox("Diabetes Mellitus", ["No", "Yes"])

    if st.button("Predict Kidney Disease"):
        values = [age, bp, sg, al, su, bgr, bu, sc, sod, pot, hemo,
                  1 if htn == "Yes" else 0, 1 if dm == "Yes" else 0]
        pred, proba = predict("kidney", values)
        if pred == 1:
            st.error(f"⚠️ High risk of Kidney Disease (confidence: {proba*100:.1f}%). "
                     "Please consult a nephrologist.")
        else:
            st.success(f"✅ Low risk of Kidney Disease (confidence: {proba*100:.1f}%). "
                       "Stay hydrated and monitor blood pressure regularly.")

# ---------------- PARKINSON'S ----------------
elif page == "Parkinson's Disease":
    st.title("🧠 Parkinson's Disease Prediction")
    st.caption("Values are derived from voice-recording measurements (as used in clinical studies).")
    c1, c2, c3 = st.columns(3)
    with c1:
        fo = st.number_input("MDVP:Fo(Hz)", 60.0, 300.0, 150.0)
        fhi = st.number_input("MDVP:Fhi(Hz)", 60.0, 400.0, 200.0)
        flo = st.number_input("MDVP:Flo(Hz)", 50.0, 300.0, 110.0)
        jitter = st.number_input("MDVP:Jitter(%)", 0.0, 0.05, 0.006, format="%.5f")
    with c2:
        shimmer = st.number_input("MDVP:Shimmer", 0.0, 0.2, 0.03, format="%.4f")
        nhr = st.number_input("NHR", 0.0, 0.5, 0.02, format="%.4f")
        hnr = st.number_input("HNR", 0.0, 40.0, 21.0)
        rpde = st.number_input("RPDE", 0.0, 1.0, 0.45)
    with c3:
        dfa = st.number_input("DFA", 0.0, 1.0, 0.7)
        spread1 = st.number_input("spread1", -10.0, 0.0, -5.6)
        spread2 = st.number_input("spread2", 0.0, 1.0, 0.22)
        d2 = st.number_input("D2", 0.0, 5.0, 2.4)
        ppe = st.number_input("PPE", 0.0, 1.0, 0.2)

    if st.button("Predict Parkinson's Disease"):
        values = [fo, fhi, flo, jitter, shimmer, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]
        pred, proba = predict("parkinsons", values)
        if pred == 1:
            st.error(f"⚠️ High risk of Parkinson's Disease (confidence: {proba*100:.1f}%). "
                     "Please consult a neurologist.")
        else:
            st.success(f"✅ Low risk of Parkinson's Disease (confidence: {proba*100:.1f}%). "
                       "No major indicators found in the voice parameters entered.")
