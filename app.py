import streamlit as st
import pandas as pd
import pickle

# Load models and feature names
pcod_model = pickle.load(open("pcod_model.pkl", "rb"))
pcos_model = pickle.load(open("pcos_model.pkl", "rb"))
pcod_features = pickle.load(open("pcod_features.pkl", "rb"))
pcos_features = pickle.load(open("pcos_features.pkl", "rb"))

# Manual inputs
pcod_manual = ["Period Length", "Cycle Length", "Age"]
pcos_manual = [
    "Age (in Years)",
    "Weight (in Kg)",
    "Height (in Cm / Feet)",
    "After how many months do you get your periods?"
]


def get_input_ui(features, manual_fields, prefix):
    inputs = {}
    for feat in features:
        key = f"{prefix}_{feat}"
        if feat in manual_fields:
            inputs[feat] = st.number_input(f"{prefix} - {feat}", key=key)
        else:
            inputs[feat] = st.radio(f"{prefix} - {feat}", [0, 1], key=key)
    return list(inputs.values())


def predict_combined(pcod_vals, pcos_vals):
    df_pcod = pd.DataFrame([pcod_vals], columns=pcod_features)
    df_pcos = pd.DataFrame([pcos_vals], columns=pcos_features)

    pcod_pred = pcod_model.predict(df_pcod)[0]
    pcod_prob = pcod_model.predict_proba(df_pcod)[0][1]

    pcos_pred = pcos_model.predict(df_pcos)[0]
    pcos_prob = pcos_model.predict_proba(df_pcos)[0][1]

    result_text = f"🔹 **PCOD Prediction**: {'Yes (1)' if pcod_pred == 1 else 'No (0)'} | Probability: {pcod_prob:.2f}\n"
    result_text += f"🔸 **PCOS Prediction**: {'Yes (1)' if pcos_pred == 1 else 'No (0)'} | Probability: {pcos_prob:.2f}\n\n"

    if pcod_prob < 0.3 and pcos_prob < 0.3:
        result_text += "✅ You are unlikely to have either PCOD or PCOS."
    elif pcod_prob >= 0.3 and pcod_prob > pcos_prob:
        result_text += "⚠️ You are more likely to have **PCOD**."
    elif pcos_prob >= 0.3 and pcos_prob > pcod_prob:
        result_text += "⚠️ You are more likely to have **PCOS**."
    else:
        result_text += "⚠️ There is a possibility of both PCOD and PCOS."

    return result_text


# UI
st.title("🩺 PCOD / PCOS Prediction App")
st.markdown("Please answer the following questions carefully:")

st.subheader("🔹 PCOD Questions")
pcod_inputs = get_input_ui(pcod_features, pcod_manual, "PCOD")

st.subheader("🔸 PCOS Questions")
pcos_inputs = get_input_ui(pcos_features, pcos_manual, "PCOS")

if st.button("🔍 Predict"):
    result = predict_combined(pcod_inputs, pcos_inputs)
    st.markdown("### 🧾 Result Summary:")
    st.info(result)
