# infant-retinal-hemorrhage-ML
# 👶 RH Score Predictor for Newborns

This app predicts Retinal Hemorrhage (RH) score based on newborn health factors using a trained machine learning model (LightGBM/XGBoost/RandomForest).

👉 [Try it Live on Streamlit](https://juahnoh-infant-retinal-hemorrhage-ml.streamlit.app)

## Features
- Gender, birth category, gestational age
- Apgar score at 1 and 5 minutes
- Outputs predicted RH Score and class probabilities

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
