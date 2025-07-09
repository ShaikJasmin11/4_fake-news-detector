# app.py

import streamlit as st
import joblib
from src.preprocess import clean_text

st.set_page_config(page_title="📰 Fake News Detector", layout="centered")

st.title("📰 Fake News Detection App")
st.write("Enter a news article below and the AI will tell you if it's **Real** or **Fake**.")

# Load model and vectorizer
model = joblib.load("models/news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Session state setup
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# Input field
st.session_state.user_input = st.text_area("Paste the news article text here:", value=st.session_state.user_input)

# Buttons
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Check"):
        cleaned = clean_text(st.session_state.user_input)
        vectorized = vectorizer.transform([cleaned])
        st.session_state.prediction = model.predict(vectorized)[0]

with col2:
    if st.button("Reset"):
        st.session_state.user_input = ""
        st.session_state.prediction = None
        st.rerun()  # The official non-experimental version

# Output
if st.session_state.prediction is not None:
    if st.session_state.prediction == 1:
        st.success("✅ This news is likely **REAL**.")
    else:
        st.error("🚨 This news is likely **FAKE**.")
