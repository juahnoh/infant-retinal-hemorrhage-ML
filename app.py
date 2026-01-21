import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import streamlit as st


st.write("PYTHON:", sys.version)

st.title("👶 RH Score Predictor for Newborns")

# --- 모델 선택 ---
model_files = {
    'RandomForest': 'RandomForest_best_pipeline_rh_score.pkl',
    'LightGBM': 'LightGBM_best_pipeline_rh_score.pkl',
    'XGBoost': 'XGBoost_best_pipeline_rh_score.pkl',
    'KNN': 'KNN_best_pipeline_rh_score.pkl'
}

selected_model = st.selectbox("🤖 Select Model", list(model_files.keys()))

# 선택된 모델 로드
try:
    pipeline = joblib.load(model_files[selected_model])
    st.success(f"✅ {selected_model} model loaded successfully!")
except FileNotFoundError:
    st.error(f"❌ Model file not found: {model_files[selected_model]}")
    st.stop()

st.markdown("---")

import traceback
try:
    pipeline = joblib.load(model_files[selected_model])
    st.success(f"✅ {selected_model} model loaded successfully!")
except Exception as e:
    st.error(f"❌ Model load failed: {type(e).__name__}: {e}")
    st.code(traceback.format_exc())
    st.stop()

# --- 사용자 입력 ---
col1, col2 = st.columns(2)

with col1:
    gender_label = st.selectbox("Gender", ["Male", "Female"])
    gender = 0 if gender_label == "Male" else 1
    
    birth_label = st.selectbox("Birth Method", ["NSVD", "C-section"])
    birth = 0 if birth_label == "NSVD" else 1
    
    ga_days = st.number_input("Gestational Age (days)", min_value=200, max_value=300, value=272)

with col2:
    wt = st.number_input("Birth Weight (g)", min_value=1000, max_value=5000, value=3200)
    
    apgar_1 = st.selectbox("Apgar Score at 1 min", list(range(1, 11)), index=6)
    
    apgar_5 = st.selectbox("Apgar Score at 5 min", list(range(1, 11)), index=8)

st.markdown("---")

# --- 예측 실행 ---
if st.button("🔮 Predict RH Score", type="primary"):
    # Feature engineering (원래 코드와 동일하게)
    apgar_diff = apgar_5 - apgar_1
    wt_per_ga = wt / (ga_days + 1e-6)
    ga_sq = ga_days ** 2
    wt_sq = wt ** 2
    
    # 입력 데이터 생성
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
    
    # 예측
    prediction = pipeline.predict(input_data)[0]
    proba = pipeline.predict_proba(input_data)[0]
    
    # 결과 표시
    st.markdown("### 📊 Prediction Results")
    
    col_result1, col_result2 = st.columns(2)
    
    with col_result1:
        st.metric("🧠 Predicted RH Score", f"{prediction}")
    
    with col_result2:
        max_prob = max(proba)
        st.metric("🎯 Confidence", f"{max_prob:.1%}")
    
    st.markdown("---")
    
    # 확률 분포 표시
    st.markdown("### 🔢 Prediction Probabilities")
    
    prob_df = pd.DataFrame({
        'RH Score': range(len(proba)),
        'Probability': proba
    })
    
    # 막대 그래프
    st.bar_chart(prob_df.set_index('RH Score'))
    
    # 테이블로도 표시
    st.dataframe(
        prob_df.style.format({'Probability': '{:.2%}'}).background_gradient(cmap='Blues'),
        use_container_width=True
    )
    
    # Severe 여부 판단 (rh_score >= 3)
    severe_prob = sum(proba[3:]) if len(proba) > 3 else 0
    
    st.markdown("---")
    st.markdown("### ⚠️ Severity Assessment")
    
    if severe_prob > 0.5:
        st.error(f"🚨 High risk of severe RH (score ≥ 3): {severe_prob:.1%}")
    else:
        st.success(f"✅ Low risk of severe RH (score ≥ 3): {severe_prob:.1%}")
