import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 1. Page config
# =========================
st.set_page_config(
    page_title="Pulmonary Hypertension Risk Calculator Based on MCG",
    page_icon="🩺",
    layout="wide",
)

# =========================
# 2. Paths / constants
# =========================
MODEL_PATH = "RandomForest_best_model.joblib"
FEATURE_FILE = "selected_features_1SE_建模数据.txt"
PH_THRESHOLD = 0.352361111

FEATURE_LABELS = {
    "sex": "Sex",
    "age": "Age (years)",
    "BMI": "BMI",
    "R_Mag_Ang": "R_Mag_Ang",
    "QRS_TCV_AREA": "QRS_TCV_AREA",
    "QTc": "QTc",
    "P_P_MFD": "P_P_MFD",
    "T_Mag_Ang": "T_Mag_Ang",
    "R_TCV_Ang": "R_TCV_Ang",
    "P_Mag_Dis": "P_Mag_Dis",
}

# =========================
# 3. Custom style
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background: #ffffff;
        color: #111827;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    h1, h2, h3, h4, h5, h6, p, label, div, span {
        color: #111827;
    }

    .title-main {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        color: #111827;
    }

    .subtitle-main {
        color: #4b5563;
        font-size: 1.02rem;
        margin-bottom: 1.2rem;
    }

    .result-high {
        background: #fff7f7;
        border-left: 5px solid #ff5a67;
        border-radius: 16px;
        padding: 18px;
        color: #111827;
        margin-bottom: 16px;
    }

    .result-low {
        background: #f0fdf4;
        border-left: 5px solid #16a34a;
        border-radius: 16px;
        padding: 18px;
        color: #111827;
        margin-bottom: 16px;
    }

    .advice-box {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 16px 18px;
        color: #111827;
        line-height: 1.65;
    }

    .shap-box {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 12px;
    }

    .small-note {
        color: #6b7280;
        font-size: 0.92rem;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 12px 14px;
        border-radius: 14px;
        color: #111827;
    }

    .stButton > button {
        background: #1d7cf2;
        color: white;
        border: none;
        border-radius: 12px;
        height: 3rem;
        font-weight: 700;
        width: 100%;
    }

    .stButton > button:hover {
        background: #1669cd;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# 4. Helpers
# =========================
@st.cache_resource
def load_model():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_features():
    if not Path(FEATURE_FILE).exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURE_FILE}")

    with open(FEATURE_FILE, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if "," in content:
        features = [x.strip() for x in content.split(",") if x.strip()]
    else:
        features = [x.strip() for x in content.splitlines() if x.strip()]

    return features



def get_processed_input_for_shap(model, input_df):
    if hasattr(model, "steps") and isinstance(model.steps, list):
        xt = input_df.copy()
        for _, step in model.steps[:-1]:
            if step is None or step == "passthrough":
                continue
            xt = step.transform(xt)

        if hasattr(xt, "toarray"):
            xt = xt.toarray()

        xt = pd.DataFrame(xt)
        estimator = model.steps[-1][1]
        return xt, estimator

    return input_df.copy(), model



def make_shap_explanation(model, input_df, feature_names):
    processed_df, final_estimator = get_processed_input_for_shap(model, input_df)
    explainer = shap.TreeExplainer(final_estimator)
    shap_obj = explainer(processed_df)

    if hasattr(shap_obj, "values"):
        if len(shap_obj.values.shape) == 3:
            values = shap_obj.values[0, :, 1]
            base_value = shap_obj.base_values[0, 1]
        else:
            values = shap_obj.values[0]
            base_value = shap_obj.base_values[0]
    else:
        raise ValueError("Unable to parse SHAP output.")

    if processed_df.shape[1] == len(feature_names):
        data_vals = input_df.iloc[0].values
        used_names = [FEATURE_LABELS.get(f, f) for f in feature_names]
    else:
        data_vals = processed_df.iloc[0].values
        used_names = [f"Feature {i+1}" for i in range(processed_df.shape[1])]

    return shap.Explanation(
        values=values,
        base_values=base_value,
        data=data_vals,
        feature_names=used_names,
    )



def plot_waterfall(explanation, max_display=12):
    fig = plt.figure(figsize=(8, 6), dpi=220)
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    return fig


# =========================
# 5. Load model/resources
# =========================
try:
    model = load_model()
    feature_names = load_features()
except Exception as e:
    st.error(f"Failed to load model/resources: {e}")
    st.stop()


# =========================
# 6. Header
# =========================
st.markdown(
    '<div class="title-main">🩺 Pulmonary Hypertension Risk Calculator Based on MCG</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle-main">Enter the patient MCG features below and click <b>Predict</b> to obtain the diagnostic probability and SHAP explanation.</div>',
    unsafe_allow_html=True,
)

st.markdown("---")


# =========================
# 7. Input section
# =========================
st.markdown("## 📋 Patient Feature Input")
st.write("Please enter the input features. Turn on SHAP explanation if you want the waterfall plot.")

show_shap = st.checkbox(
    "Show SHAP explanation (slower)",
    value=True,
    key="show_shap_main",
)

input_data = {}
cols_per_row = 5
n_rows = int(np.ceil(len(feature_names) / cols_per_row))

for r in range(n_rows):
    row_cols = st.columns(cols_per_row)
    for c in range(cols_per_row):
        idx = r * cols_per_row + c
        if idx >= len(feature_names):
            break

        feat = feature_names[idx]
        label = FEATURE_LABELS.get(feat, feat)

        with row_cols[c]:
            if feat.lower() in ["sex", "gender"]:
                sex_label = st.selectbox(
                    label,
                    ["Female", "Male"],
                    index=0,
                    key=f"select_{feat}",
                )
                input_data[feat] = 0 if sex_label == "Female" else 1

            elif feat == "age":
                input_data[feat] = st.number_input(
                    label,
                    min_value=0,
                    value=30,
                    step=1,
                    format="%d",
                    key=f"num_{feat}",
                )

            else:
                input_data[feat] = st.number_input(
                    label,
                    value=0.0,
                    format="%.2f",
                    key=f"num_{feat}",
                )

input_df = pd.DataFrame([input_data], columns=feature_names)

if "age" in input_df.columns:
    input_df["age"] = input_df["age"].astype(int)

predict_clicked = st.button(
    "Predict",
    use_container_width=True,
    key="predict_button_main",
)


# =========================
# 8. Prediction
# =========================
if predict_clicked:
    try:
        prob = float(model.predict_proba(input_df)[0, 1])
    except Exception:
        pred_raw = int(model.predict(input_df)[0])
        prob = 1.0 if pred_raw == 1 else 0.0

    pred = 1 if prob >= PH_THRESHOLD else 0

    shap_explanation = None
    shap_ok = False

    if show_shap:
        try:
            shap_explanation = make_shap_explanation(model, input_df, feature_names)
            shap_ok = True
        except Exception as e:
            st.warning(f"SHAP explanation failed: {e}")
            shap_ok = False

    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown("## 📊 Prediction Result")

        if pred == 1:
            st.markdown(
                """
                <div class="result-high">
                    <div style="font-size:1.9rem;font-weight:800;color:#ef4444;">⚠️ High Risk of Pulmonary Hypertension</div>
                    <div style="margin-top:10px;color:#6b7280;font-size:0.98rem;">
                        This result is provided for research and decision-support purposes only.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="result-low">
                    <div style="font-size:1.9rem;font-weight:800;color:#16a34a;">✅ Low Risk of Pulmonary Hypertension</div>
                    <div style="margin-top:10px;color:#6b7280;font-size:0.98rem;">
                        This result is provided for research and decision-support purposes only.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("## 🩺 Decision Support")
        if pred == 1:
            st.markdown(
                """
                <div class="advice-box">
                    The MCG model indicates that the patient has a high probability of pulmonary hypertension.<br><br>
                    <b>Suggested next step:</b> referral to a pulmonary vascular center for further TTE or RHC examination.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="advice-box">
                    The model indicates a relatively low probability of pulmonary hypertension at present.<br><br>
                    <b>Suggested next step:</b> continue routine follow-up.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right_col:
        st.markdown("## 🔎 SHAP Explainability Analysis")
        st.write(
            "The waterfall plot illustrates how each feature contributes to the current prediction. "
            "Red features push the model toward higher risk, while blue features push it toward lower risk."
        )

        if show_shap and shap_ok and shap_explanation is not None:
            st.markdown('<div class="shap-box">', unsafe_allow_html=True)
            fig = plot_waterfall(shap_explanation, max_display=12)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        elif show_shap:
            st.info("SHAP plot is unavailable for this prediction.")
        else:
            st.info("Enable SHAP explanation above to display the waterfall plot.")

    st.markdown("---")
    st.markdown(
        f'<div class="small-note">This tool is for research and decision-support only.</div>',
        unsafe_allow_html=True,
    )
