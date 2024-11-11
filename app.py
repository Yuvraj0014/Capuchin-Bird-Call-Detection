import streamlit as st
import os
import librosa
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('audio_classification_model.h5')

# Set up the upload directory
UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Preprocess audio file
def preprocess_audio(file_path, target_shape=(1491, 257)):
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
    mel_spectrogram = np.resize(mel_spectrogram, target_shape)
    return np.array([mel_spectrogram])

# Streamlit App
st.title("🔊 Capuchin Bird Call Detection")
st.write("""
    Detect the presence of Capuchin bird calls in your audio files using advanced deep learning.
    This tool is particularly useful for wildlife monitoring and research on Capuchin bird habitats.
""")

# Instructions
with st.expander("ℹ️ Instructions"):
    st.write("""
        - **Supported Formats**: Upload audio files in WAV or MP3 format.
        - **Confidence Levels**: Each file is analyzed, and a confidence score is provided.
        - **Result**: A message will indicate whether a Capuchin bird call was detected, with a confidence level.
    """)

# File uploader widget
uploaded_files = st.file_uploader("📂 Upload your audio files here", type=['wav', 'mp3'], accept_multiple_files=True)

if uploaded_files:
    # Initialize result storage
    result_summary = []

    # Processing files
    for file in uploaded_files:
        st.markdown(f"#### Processing file: {file.name}")
        file_path = os.path.join(UPLOAD_FOLDER, file.name)
        
        # Save uploaded file locally
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        # Display audio player
        st.audio(file)

        # Preprocess the file and make prediction
        input_data = preprocess_audio(file_path)
        predictions = model.predict(input_data)
        confidence = predictions[0][0]
        
        # Result determination
        if confidence >= 0.5:
            confidence_level = f"{confidence * 100:.2f}%"
            st.markdown(
                f"<span style='color:#006400; font-weight:bold;'>CAPUCHIN BIRD CALL FOUND "
                f"(Confidence: {confidence_level})</span>", unsafe_allow_html=True
            )
            result_summary.append(f"{file.name}: Capuchin Bird Call Found with {confidence_level} confidence.")
        else:
            confidence_level = f"{(1 - confidence) * 100:.2f}%"
            st.markdown(
                f"<span style='color:#8B0000; font-weight:bold;'>CAPUCHIN BIRD CALL NOT FOUND "
                f"(Confidence: {confidence_level})</span>", unsafe_allow_html=True
            )
            result_summary.append(f"{file.name}: No Capuchin Bird Call Found with {confidence_level} confidence.")

    # Display summary of results after all files are processed
    st.markdown("### 📊 Summary of Results")
    for result in result_summary:
        st.write(result)

    # Completion message
    st.success("Processing complete! All files have been analyzed.")
else:
    st.info("📌 Please upload one or more audio files to begin analysis.")
