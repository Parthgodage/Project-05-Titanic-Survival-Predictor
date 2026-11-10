
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Titanic AI Predictor",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 Titanic Survival Predictor")
st.write("Predict passenger survival using a Machine Learning model.")

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

st.markdown("### Passenger Details")

pclass = st.selectbox("Ticket Class", [1, 2, 3])
sex = st.radio("Gender", ["Female", "Male"], horizontal=True)
age = st.slider("Age", 1, 80, 25)
fare = st.slider("Ticket Fare", 0.0, 500.0, 32.0, 1.0)

st.markdown("---")

if st.button("Predict Survival", use_container_width=True):
    try:
        if not Path("model.joblib").exists():
            st.error("Model file is missing. Please train the model first.")
        else:
            model = load_model()

            passenger = pd.DataFrame([{
                "Pclass": pclass,
                "Sex": sex.lower(),
                "Age": age,
                "Fare": fare
            }])

            prediction = model.predict(passenger)[0]
            probabilities = model.predict_proba(passenger)[0]
            survival_chance = probabilities[list(model.classes_).index(1)]

            st.subheader("Prediction Result")

            if prediction == 1:
                st.success("This passenger is predicted to survive.")
            else:
                st.error("This passenger is predicted not to survive.")

            st.metric(
                "Estimated Survival Probability",
                f"{survival_chance:.1%}"
            )

            st.progress(float(survival_chance))

    except Exception as e:
        st.error(f"Prediction could not be completed: {e}")

st.caption(
    "Educational project using historical Titanic passenger data. "
    "Predictions are estimates, not guarantees."
)
