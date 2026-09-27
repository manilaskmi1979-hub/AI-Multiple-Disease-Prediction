# AI-Based Multiple Disease Prediction
**Predicting Health, Protecting Lives**

A machine learning system that predicts the risk of **5 diseases** — Diabetes, Heart
Disease, Liver Disease, Kidney Disease, and Parkinson's Disease — from a user's
health details, using **Random Forest Classification**.

## 📁 Project Structure
```
disease_prediction_project/
├── data_generator.py     # Creates synthetic training datasets (data/*.csv)
├── train_models.py       # Trains a Random Forest model per disease
├── app.py                # Streamlit web app (user interface)
├── requirements.txt      # Python dependencies
├── data/                 # Generated/placed datasets (CSV)
└── models/               # Saved trained models (.pkl)
```

## ⚙️ How It Works (Modules)
1. **User Registration & Login** – (add auth if deploying publicly)
2. **Input Symptoms & Medical Data** – user enters health details via the form
3. **Data Processing & Preprocessing** – values scaled with `StandardScaler`
4. **Disease Prediction** – `RandomForestClassifier` predicts the possible disease
5. **Prediction Result & Suggestions** – shows result + a basic recommendation

## 🚀 Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Only if data/ is empty) generate the training datasets
python data_generator.py

# 3. Train all 5 Random Forest models
python train_models.py

# 4. Launch the web app
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`. Use the sidebar to
pick a disease, fill in the health parameters, and click **Predict**.

## 📝 Using REAL Datasets (recommended for your final project report)
This project ships with synthetic data so it runs fully offline out of the box.
For your actual submission, replace the files in `data/` with the original
public datasets (same column names are already used, so no code changes needed):

| Disease    | File            | Suggested Source                          |
|------------|-----------------|--------------------------------------------|
| Diabetes   | diabetes.csv    | Pima Indians Diabetes Dataset (Kaggle)     |
| Heart      | heart.csv       | UCI Heart Disease / Cleveland Dataset      |
| Liver      | liver.csv       | Indian Liver Patient Dataset (ILPD)        |
| Kidney     | kidney.csv      | Chronic Kidney Disease Dataset (UCI)       |
| Parkinsons | parkinsons.csv  | UCI Parkinson's Voice Dataset              |

After replacing the CSVs, just re-run `python train_models.py`.

## 🧠 Algorithm — Random Forest Classification
1. Collect Dataset – disease-related health records
2. Preprocess Data – clean, scale, split train/test
3. Train Model – build multiple decision trees on the training data
4. Make Prediction – each tree votes on the possible disease
5. Final Result – majority voting gives the final prediction

## 🔮 Future Enhancements
- Add more diseases for prediction
- Improve accuracy with advanced ML models (XGBoost, Neural Networks)
- Integrate wearable devices for real-time health monitoring
- Develop a mobile application for easy access
- Provide personalized health suggestions

## ⚠️ Disclaimer
This tool provides a quick ML-based **risk estimate only** and is **not** a
medical diagnosis. Always consult a qualified doctor for confirmation and
treatment.

---
**Project:** AI-Based Multiple Disease Prediction
**Name:** Priyadharshini B | **Reg No:** C4S27603
**Guide:** Dr. M. Gokiladevi, MCA., B.Ed., M.Phil., Ph.D.
