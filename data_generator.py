"""
data_generator.py
------------------
Generates realistic SYNTHETIC datasets (rule-based + noise) for the 5 diseases
covered by the AI-Based Multiple Disease Prediction project:
    1. Diabetes
    2. Heart Disease
    3. Liver Disease
    4. Kidney Disease
    5. Parkinson's Disease

Why synthetic? In offline / classroom settings you may not always have internet
access to pull the original UCI/Kaggle CSVs (Pima Diabetes, Cleveland Heart,
Indian Liver Patient, CKD, Parkinsons Voice datasets). This script builds
statistically-similar data (same column names, same rough value ranges, and a
real underlying relationship between features and the target) so the Random
Forest models learn genuine patterns.

If you already have the ORIGINAL CSV files (recommended for your final year
report), just drop them into the data/ folder with the same names below and
this script will be skipped automatically by train_models.py.
"""

import numpy as np
import pandas as pd
import os

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)


def save(df, name):
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    print(f"  saved {name}  -> shape {df.shape}")


def make_diabetes(n=800):
    age = np.random.randint(18, 80, n)
    pregnancies = np.clip(np.random.poisson(2, n), 0, 15)
    glucose = np.random.normal(120, 30, n).clip(60, 250)
    bp = np.random.normal(72, 12, n).clip(40, 130)
    skin = np.random.normal(23, 10, n).clip(5, 60)
    insulin = np.random.normal(85, 60, n).clip(0, 400)
    bmi = np.random.normal(30, 7, n).clip(15, 55)
    dpf = np.random.uniform(0.05, 2.4, n)

    risk = (
        0.08 * (glucose - 100) +
        0.12 * (bmi - 25) +
        0.04 * (age - 30) +
        2.0 * dpf +
        0.2 * pregnancies +
        np.random.normal(0, 2.0, n)
    )
    outcome = (risk > np.percentile(risk, 65)).astype(int)

    df = pd.DataFrame({
        "Pregnancies": pregnancies, "Glucose": glucose.round(1), "BloodPressure": bp.round(1),
        "SkinThickness": skin.round(1), "Insulin": insulin.round(1), "BMI": bmi.round(1),
        "DiabetesPedigreeFunction": dpf.round(3), "Age": age, "Outcome": outcome
    })
    return df


def make_heart(n=800):
    age = np.random.randint(29, 80, n)
    sex = np.random.randint(0, 2, n)  # 1=male 0=female
    cp = np.random.randint(0, 4, n)   # chest pain type
    trestbps = np.random.normal(131, 17, n).clip(90, 200)
    chol = np.random.normal(246, 51, n).clip(120, 450)
    fbs = np.random.binomial(1, 0.15, n)
    restecg = np.random.randint(0, 3, n)
    thalach = np.random.normal(150, 23, n).clip(70, 210)
    exang = np.random.binomial(1, 0.33, n)
    oldpeak = np.random.exponential(1.0, n).clip(0, 6.2)
    slope = np.random.randint(0, 3, n)
    ca = np.random.randint(0, 4, n)
    thal = np.random.randint(0, 3, n)

    risk = (
        0.04 * (age - 50) + 0.7 * sex + 0.6 * cp +
        0.02 * (trestbps - 130) + 0.01 * (chol - 240) +
        0.8 * exang + 0.9 * oldpeak + 0.5 * ca +
        -0.03 * (thalach - 150) +
        np.random.normal(0, 2.0, n)
    )
    target = (risk > np.percentile(risk, 55)).astype(int)

    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps.round(1), "chol": chol.round(1),
        "fbs": fbs, "restecg": restecg, "thalach": thalach.round(1), "exang": exang,
        "oldpeak": oldpeak.round(2), "slope": slope, "ca": ca, "thal": thal, "target": target
    })
    return df


def make_liver(n=700):
    age = np.random.randint(4, 90, n)
    gender = np.random.randint(0, 2, n)  # 1=male 0=female
    tb = np.random.exponential(1.5, n).clip(0.1, 20)      # total bilirubin
    db = (tb * np.random.uniform(0.2, 0.6, n)).clip(0.05, 10)
    alkphos = np.random.normal(290, 150, n).clip(60, 1200)
    sgpt = np.random.exponential(40, n).clip(5, 900)       # Alamine aminotransferase
    sgot = np.random.exponential(45, n).clip(5, 900)       # Aspartate aminotransferase
    tp = np.random.normal(6.5, 1.0, n).clip(3, 10)
    alb = np.random.normal(3.1, 0.8, n).clip(1, 6)
    agr = (alb / (tp - alb + 0.1)).clip(0.1, 3)

    risk = (
        0.5 * tb + 0.4 * db + 0.004 * alkphos + 0.01 * sgpt + 0.01 * sgot -
        0.4 * alb - 0.3 * agr + 0.01 * (age - 40) +
        np.random.normal(0, 1.5, n)
    )
    dataset = (risk > np.percentile(risk, 55)).astype(int)  # 1 = liver disease

    df = pd.DataFrame({
        "Age": age, "Gender": gender, "Total_Bilirubin": tb.round(2), "Direct_Bilirubin": db.round(2),
        "Alkaline_Phosphotase": alkphos.round(1), "Alamine_Aminotransferase": sgpt.round(1),
        "Aspartate_Aminotransferase": sgot.round(1), "Total_Protiens": tp.round(2),
        "Albumin": alb.round(2), "Albumin_and_Globulin_Ratio": agr.round(2), "Dataset": dataset
    })
    return df


def make_kidney(n=700):
    age = np.random.randint(2, 90, n)
    bp = np.random.normal(76, 14, n).clip(50, 180)
    sg = np.random.choice([1.005, 1.010, 1.015, 1.020, 1.025], n)  # specific gravity
    al = np.random.randint(0, 5, n)   # albumin
    su = np.random.randint(0, 5, n)   # sugar
    bgr = np.random.normal(148, 70, n).clip(60, 490)   # blood glucose random
    bu = np.random.normal(57, 45, n).clip(1, 300)      # blood urea
    sc = np.random.exponential(1.5, n).clip(0.2, 15)   # serum creatinine
    sod = np.random.normal(137, 8, n).clip(110, 160)
    pot = np.random.normal(4.6, 2.5, n).clip(2, 12)
    hemo = np.random.normal(12.5, 2.8, n).clip(3, 18)  # hemoglobin
    htn = np.random.binomial(1, 0.35, n)  # hypertension
    dm = np.random.binomial(1, 0.3, n)    # diabetes mellitus

    risk = (
        0.03 * (bu) + 0.6 * sc + 0.4 * al + 0.02 * (bp - 76) -
        0.3 * hemo + 0.8 * htn + 0.6 * dm - 100 * (sg - 1.02) +
        np.random.normal(0, 2.0, n)
    )
    classification = (risk > np.percentile(risk, 55)).astype(int)  # 1 = ckd

    df = pd.DataFrame({
        "age": age, "bp": bp.round(1), "sg": sg, "al": al, "su": su, "bgr": bgr.round(1),
        "bu": bu.round(1), "sc": sc.round(2), "sod": sod.round(1), "pot": pot.round(2),
        "hemo": hemo.round(2), "htn": htn, "dm": dm, "classification": classification
    })
    return df


def make_parkinsons(n=600):
    fo = np.random.normal(154, 40, n).clip(80, 260)     # MDVP:Fo(Hz)
    fhi = fo + np.random.normal(60, 30, n).clip(5, 300)
    flo = (fo - np.random.normal(30, 20, n)).clip(60, 240)
    jitter = np.random.exponential(0.006, n).clip(0.0005, 0.03)
    shimmer = np.random.exponential(0.03, n).clip(0.005, 0.15)
    nhr = np.random.exponential(0.02, n).clip(0.0005, 0.3)
    hnr = np.random.normal(21, 4, n).clip(8, 33)
    rpde = np.random.uniform(0.25, 0.7, n)
    dfa = np.random.uniform(0.55, 0.83, n)
    spread1 = np.random.normal(-5.6, 1.2, n)
    spread2 = np.random.normal(0.22, 0.08, n)
    d2 = np.random.normal(2.4, 0.4, n)
    ppe = np.random.uniform(0.04, 0.5, n)

    risk = (
        30 * jitter + 10 * shimmer + 8 * nhr - 0.15 * hnr +
        3 * rpde + 4 * dfa - 0.3 * spread1 + 6 * spread2 + 3 * ppe +
        np.random.normal(0, 1.0, n)
    )
    status = (risk > np.percentile(risk, 50)).astype(int)  # 1 = parkinsons

    df = pd.DataFrame({
        "MDVP:Fo(Hz)": fo.round(2), "MDVP:Fhi(Hz)": fhi.round(2), "MDVP:Flo(Hz)": flo.round(2),
        "MDVP:Jitter(%)": jitter.round(5), "MDVP:Shimmer": shimmer.round(4), "NHR": nhr.round(4),
        "HNR": hnr.round(2), "RPDE": rpde.round(4), "DFA": dfa.round(4), "spread1": spread1.round(3),
        "spread2": spread2.round(3), "D2": d2.round(3), "PPE": ppe.round(4), "status": status
    })
    return df


if __name__ == "__main__":
    print("Generating synthetic datasets ...")
    save(make_diabetes(), "diabetes.csv")
    save(make_heart(), "heart.csv")
    save(make_liver(), "liver.csv")
    save(make_kidney(), "kidney.csv")
    save(make_parkinsons(), "parkinsons.csv")
    print("Done. CSVs are in the data/ folder.")
