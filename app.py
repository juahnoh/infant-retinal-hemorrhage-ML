# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model pipeline
pipeline = joblib.load('trained_model.pkl')

st.title("👶 RH Score Prediction for Newborns")

# Input form
gender = st.selectbox("Gender", [0, 1])
birth = st.selectbox("Birth Category", [0, 1])
ga_days = st.number_input("Gestational Age (days)", 200, 300, value=272)
wt = st.number_input("Birth Weight (g)", 1000, 5000, value=3200)
apgar_1 = st.selectbox("Apgar Score at 1 min", list(range(1, 11)))
apgar_5 = st.selectbox("Apgar Score at 5 min", list(range(1, 11)))

if st.button("Predict RH Score"):
    apgar_diff = apgar_5 - apgar_1
    wt_per_ga = wt / (ga_days + 1e-6)
    ga_sq = ga_days ** 2
    wt_sq = wt ** 2

    input_data = pd.DataFrame([{
        'gender': gender,
        'birth': birth,
        'ga_days': ga_days,
        'wt': wt,
        'apgar_1': apgar_1,
        'apgar_5': apgar_5,
        'apgar_diff': apgar_diff,
        'wt_per_ga': wt_per_ga,
        'ga_sq': ga_sq,
        'wt_sq': wt_sq
    }])

    prediction = pipeline.predict(input_data)[0]
    proba = pipeline.predict_proba(input_data)[0]

    st.success(f"Predicted RH Score: {prediction}")
    st.write("Prediction Probabilities:")
    for i, p in enumerate(proba):
        st.write(f"Score {i}: {p:.2%}")
