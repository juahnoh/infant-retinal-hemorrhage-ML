import sys
import traceback
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RH Score Predictor", page_icon="👶", layout="centered")

st.write("PYTHON:", sys.version)
st.title("👶 RH Score Predictor for Newborns")
st.caption("Predict retinal hemorrhage (RH) score from basic birth parameters.")

st.markdown("---")

# --- 모델 선택 ---
model_files = {
    "LightGBM": "LightGBM_best_pipeline_rh_score.pkl",
    "XGBoost": "XGBoost_best_pipeline_rh_score.pkl",
    "KNN": "KNN_best_pipeline_rh_score.pkl",
}

selected_model = st.selectbox("🤖 Select Model", list(model_files.keys()))

# --- 모델 로드 (한 번만) ---
model_path = model_files[selected_model]
try:
    pipeline = joblib.load(model_path)
    st.success(f"✅ {selected_model} model loaded successfully!")
except FileNotFoundError:
    st.error(f"❌ Model file not found: {model_path}")
    st.stop()
except Exception as e:
    st.error(f"❌ Model load failed: {type(e).__name__}: {e}")
    st.code(traceback.format_exc())
    st.stop()

st.markdown("---")

# --- 사용자 입력 ---
col1, col2 = st.columns(2)

with col1:
    gender_label = st.selectbox("Gender", ["Male", "Female"])
    birth_label = st.selectbox("Birth Method", ["NSVD", "C-section"])
    ga_days = st.number_input("Gestational Age (days)", min_value=200, max_value=320, value=272, step=1)

with col2:
    wt = st.number_input("Birth Weight (g)", min_value=500, max_value=6000, value=3200, step=10)
    apgar_1 = st.selectbox("Apgar Score at 1 min", list(range(0, 11)), index=6)
    apgar_5 = st.selectbox("Apgar Score at 5 min", list(range(0, 11)), index=8)

st.markdown("---")

# --- 예측 실행 ---
if st.button("🔮 Predict RH Score", type="primary"):
    # Feature engineering
    apgar_diff = apgar_5 - apgar_1
    wt_per_ga = wt / (ga_days + 1e-6)
    ga_sq = float(ga_days) ** 2
    wt_sq = float(wt) ** 2

    # ✅ 중요: 범주형은 "라벨 문자열" 그대로 넣음 (OHE/ColumnTransformer 호환)
    input_data = pd.DataFrame([{
        "gender": str(gender_label),
        "birth": str(birth_label),
        "ga_days": float(ga_days),
        "wt": float(wt),
        "apgar_1": int(apgar_1),
        "apgar_5": int(apgar_5),
        "apgar_diff": int(apgar_diff),
        "wt_per_ga": float(wt_per_ga),
        "ga_sq": float(ga_sq),
        "wt_sq": float(wt_sq),
    }])

    # 공백 문자열/빈 값 방어
    input_data = input_data.replace(r"^\s*$", np.nan, regex=True)

    # 파이프라인이 기대하는 컬럼에 맞춰 정렬/보정 (있으면)
    if hasattr(pipeline, "feature_names_in_"):
        expected = list(pipeline.feature_names_in_)
        for c in expected:
            if c not in input_data.columns:
                input_data[c] = np.nan
        input_data = input_data[expected]

    # (선택) 디버그 출력 토글
    with st.expander("🛠️ Debug (optional)"):
        st.write("input_data:")
        st.dataframe(input_data)
        st.write("dtypes:")
        st.write(input_data.dtypes)
        if hasattr(pipeline, "feature_names_in_"):
            st.write("pipeline.feature_names_in_:")
            st.write(list(pipeline.feature_names_in_))

    # --- 예측 ---
    try:
        prediction = pipeline.predict(input_data)[0]

        # predict_proba 없을 수 있으니 방어
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(input_data)[0]
        else:
            proba = None

    except Exception as e:
        st.error(f"❌ Predict failed: {type(e).__name__}: {e}")
        st.code(traceback.format_exc())
        st.stop()

    # --- 결과 표시 ---
    st.markdown("### 📊 Prediction Results")
    col_result1, col_result2 = st.columns(2)

    with col_result1:
        st.metric("🧠 Predicted RH Score", f"{prediction}")

    with col_result2:
        if proba is not None:
            st.metric("🎯 Confidence", f"{float(np.max(proba)):.1%}")
        else:
            st.metric("🎯 Confidence", "N/A")

    # --- 확률 분포 ---
    if proba is not None:
        st.markdown("---")
        st.markdown("### 🔢 Prediction Probabilities")

        prob_df = pd.DataFrame({
            "RH Score": list(range(len(proba))),
            "Probability": proba
        })

        st.bar_chart(prob_df.set_index("RH Score"))
        st.dataframe(
            prob_df.style.format({"Probability": "{:.2%}"}).background_gradient(cmap="Blues"),
            use_container_width=True
        )

        # Severe 여부 판단 (rh_score >= 3)
        severe_prob = float(np.sum(proba[3:])) if len(proba) > 3 else 0.0

        st.markdown("---")
        st.markdown("### ⚠️ Severity Assessment")

        if severe_prob > 0.5:
            st.error(f"🚨 High risk of severe RH (score ≥ 3): {severe_prob:.1%}")
        else:
            st.success(f"✅ Low risk of severe RH (score ≥ 3): {severe_prob:.1%}")
    else:
        st.info("This model does not support predict_proba(), so probability outputs are not available.")
