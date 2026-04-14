import streamlit as st
import numpy as np
import pickle

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Cancer Risk Predictor",
    page_icon="🫁",
    layout="centered"
)

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_model()

# ── UI ───────────────────────────────────────────────────────
st.title("🫁 Cancer Risk Level Predictor")
st.markdown("Remplissez les informations du patient pour prédire le niveau de risque.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Informations générales")
    age    = st.slider("Âge", 10, 80, 35)
    gender = st.selectbox("Genre", ["Homme", "Femme"])
    gender_val = 1 if gender == "Homme" else 2

    st.subheader("🌍 Facteurs environnementaux")
    air_pollution       = st.slider("Air Pollution",        1, 8, 3)
    dust_allergy        = st.slider("Dust Allergy",         1, 8, 3)
    occupational_hazard = st.slider("Occupational Hazards", 1, 8, 3)

    st.subheader("🧬 Facteurs génétiques / médicaux")
    genetic_risk        = st.slider("Genetic Risk",         1, 8, 3)
    chronic_lung        = st.slider("Chronic Lung Disease", 1, 8, 3)

with col2:
    st.subheader("🍺 Mode de vie")
    alcohol       = st.slider("Alcohol Use",      1, 8, 2)
    balanced_diet = st.slider("Balanced Diet",    1, 8, 5)
    obesity       = st.slider("Obesity",          1, 8, 3)
    smoking       = st.slider("Smoking",          1, 8, 2)
    passive_smoker= st.slider("Passive Smoker",   1, 8, 2)

    st.subheader("🤒 Symptômes")
    chest_pain      = st.slider("Chest Pain",           1, 9, 3)
    coughing_blood  = st.slider("Coughing of Blood",    1, 9, 2)
    fatigue         = st.slider("Fatigue",              1, 9, 3)
    weight_loss     = st.slider("Weight Loss",          1, 9, 3)
    shortness       = st.slider("Shortness of Breath",  1, 9, 3)
    wheezing        = st.slider("Wheezing",             1, 8, 2)
    swallowing      = st.slider("Swallowing Difficulty",1, 8, 2)
    clubbing        = st.slider("Clubbing of Finger Nails",1,9,3)
    frequent_cold   = st.slider("Frequent Cold",        1, 7, 2)
    dry_cough       = st.slider("Dry Cough",            1, 7, 2)
    snoring         = st.slider("Snoring",              1, 7, 2)

st.divider()

# ── Prediction ───────────────────────────────────────────────
if st.button("🔍 Prédire le niveau de risque", use_container_width=True, type="primary"):

    features = np.array([[
        age, gender_val, air_pollution, alcohol, dust_allergy,
        occupational_hazard, genetic_risk, chronic_lung, balanced_diet,
        obesity, smoking, passive_smoker, chest_pain, coughing_blood,
        fatigue, weight_loss, shortness, wheezing, swallowing,
        clubbing, frequent_cold, dry_cough, snoring
    ]])

    pred       = model.predict(features)[0]
    proba      = model.predict_proba(features)[0]
    label      = le.inverse_transform([pred])[0]
    confidence = proba.max() * 100

    # Result card
    if label == "High":
        st.error(f"🔴 Niveau de risque : **HIGH** — Confiance : {confidence:.1f}%")
        st.warning("⚠️ Risque élevé détecté. Une consultation médicale urgente est recommandée.")
    elif label == "Medium":
        st.warning(f"🟡 Niveau de risque : **MEDIUM** — Confiance : {confidence:.1f}%")
        st.info("ℹ️ Risque modéré. Un suivi médical est conseillé.")
    else:
        st.success(f"🟢 Niveau de risque : **LOW** — Confiance : {confidence:.1f}%")
        st.info("✅ Risque faible. Continuez à maintenir un mode de vie sain.")

    # Probabilities
    st.subheader("📊 Probabilités par classe")
    classes = le.classes_
    prob_df = {c: f"{p*100:.1f}%" for c, p in zip(classes, proba)}
    cols = st.columns(3)
    colors = {"High": "🔴", "Low": "🟢", "Medium": "🟡"}
    for i, (c, p) in enumerate(zip(classes, proba)):
        cols[i].metric(label=f"{colors[c]} {c}", value=f"{p*100:.1f}%")

st.divider()
st.caption("🎓 Projet Machine Learning — Modèle : Random Forest (Accuracy: 98.3%)")
