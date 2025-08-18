import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 모델 불러오기
pipeline = joblib.load('trained_model_pipeline.pkl')

st.title("👶 RH Score Predictor for Newborns")

# --- 🔽 사용자 입력 (라벨은 보기 좋게, 값은 숫자로 변환) ---
gender_label = st.selectbox("Gender", ["Female", "Male"])
gender = 0 if gender_label == "Male" else 1

birth_label = st.selectbox("Birth Method", ["NSVD", "C-section"])
birth = 0 if birth_label == "NSVD" else 1

ga_days = st.number_input("Gestational Age (days)", min_value=200, max_value=300, value=272)
wt = st.number_input("Birth Weight (g)", min_value=1000, max_value=5000, value=3200)
apgar_1 = st.selectbox("Apgar Score at 1 min", list(range(1, 11)))
apgar_5 = st.selectbox("Apgar Score at 5 min", list(range(1, 11)))

# --- 🔮 예측 실행 ---
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

    st.success(f"🧠 Predicted RH Score: {prediction}")
    st.write("🔢 Prediction Probabilities:")
    for i, p in enumerate(proba):
        st.write(f"Score {i}: {p:.2%}")
