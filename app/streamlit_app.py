import streamlit as st
import joblib
import os
import numpy as np

MODEL_PATH = "../model/model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("Thread Escalation Predictor")

thread_text = st.text_area("Paste conversation thread:")

if st.button("Analyze"):

    model = load_model()

    messages = [m.strip() for m in thread_text.split("\n") if m.strip()]

    if len(messages) < 2:
        st.warning("Need at least 2 messages")
        st.stop()

    scores = model.predict_proba(messages)[:, 1]

    risk = float(np.mean(scores)) * 100

    st.metric("Escalation Risk", f"{risk:.1f}%")
