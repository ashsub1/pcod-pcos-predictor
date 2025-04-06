import streamlit as st
import pickle
import pandas as pd

# Load models and features
pcod_model = pickle.load(open("pcod_model.pkl", "rb"))
pcod_features = pickle.load(open("pcod_features.pkl", "rb"))

pcos_model = pickle.load(open("pcos_model.pkl", "rb"))
pcos_features = pickle.load(open("pcos_features.pkl", "rb"))

st.set_page_config(page_title="PCOD / PCOS Prediction Web", layout="centered")

st.title("🩺 PCOD / PCOS Prediction Web")
st.markdown("""
**Instructions**  
- For Yes/No questions, use `1` for **Yes** and `0` for **No**.  
- Enter values carefully for accurate results.
""")

# Keywords to detect integer (number_input) fields
manual_keywords = ["age", "weight", "height", "length", "months", "period"]

# Forced binary fields (regardless of keyword detection)
force_binary_fields = [
    "overweight",
    "weight loss or weight gain",
    "irregular or missed periods"
]

# Determine whether a feature should be an integer input
def is_integer_input(feature_name):
    fname = feature_name.lower().strip()
    # If the feature contains any of the forced binary keywords, force binary input
    for fb in force_binary_fields:
        if fb in fname:
            return False
    # Otherwise, if any manual keyword is found, use integer input
    return any(kw in fname for kw in manual_keywords)

# Input generation function
def get_input(features, key_prefix):
    inputs = {}
    for feat in features:
        key = f"{key_prefix}_{feat}"
        if is_integer_input(feat):
            inputs[feat] = st.number_input(feat, min_value=0, step=1, format="%d", key=key)
        else:
            inputs[feat] = st.radio(feat, [0, 1], horizontal=True, key=key)
    return inputs

# Input sections
st.subheader("🔹 PCOD Input")
pcod_input = get_input(pcod_features, key_prefix="pcod")

st.subheader("🔸 PCOS Input")
pcos_input = get_input(pcos_features, key_prefix="pcos")

# Prediction
if st.button("🔍 Predict"):
    df_pcod = pd.DataFrame([pcod_input])
    df_pcos = pd.DataFrame([pcos_input])

    pcod_pred = pcod_model.predict(df_pcod)[0]
    pcod_prob = pcod_model.predict_proba(df_pcod)[0][1]

    pcos_pred = pcos_model.predict(df_pcos)[0]
    pcos_prob = pcos_model.predict_proba(df_pcos)[0][1]

    st.markdown("### 🧾 Prediction Summary")
    st.write(f"🔹 **PCOD Prediction**: {'1 (Yes)' if pcod_pred == 1 else '0 (No)'} | Probability: `{pcod_prob:.2f}`")
    st.write(f"🔸 **PCOS Prediction**: {'1 (Yes)' if pcos_pred == 1 else '0 (No)'} | Probability: `{pcos_prob:.2f}`")

    # Diagnosis suggestion based on probabilities
    if pcod_prob < 0.3 and pcos_prob < 0.3:
        st.success("✅ You are unlikely to have either PCOD or PCOS.")
    elif pcod_prob >= 0.3 and pcod_prob > pcos_prob:
        st.warning("⚠️ You are more likely to have **PCOD**.")
    elif pcos_prob >= 0.3 and pcos_prob > pcod_prob:
        st.warning("⚠️ You are more likely to have **PCOS**.")
    else:
        st.warning("⚠️ There is a possibility of both PCOD and PCOS.")
