import streamlit as st
import librosa
import numpy as np
import joblib
import os

# Load model and label encoder
model = joblib.load("siren_model.pkl")
le = joblib.load("label_encoder.pkl")

st.title("🚨 Urban Micro-Siren Detector")
st.markdown("Upload a `.wav` file to detect sirens: ambulance, police, firetruck, or traffic.")

# File uploader
audio_file = st.file_uploader("Upload an audio file", type=["wav"])

if audio_file:
    # Save file temporarily
    with open("temp.wav", "wb") as f:
        f.write(audio_file.read())

    # Extract MFCC features
    try:
        y, sr = librosa.load("temp.wav", sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1).reshape(1, -1)

        # Predict
        prediction = model.predict(mfcc_mean)[0]
        class_name = le.inverse_transform([prediction])[0]
        st.success(f"🧠 Prediction: **{class_name.upper()}**")

    except Exception as e:
        st.error(f"Error processing audio: {e}")
