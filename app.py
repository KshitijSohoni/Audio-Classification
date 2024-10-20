import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.models import load_model
import io

model = load_model('C:/Users/kshit/Desktop/DL Lab Miniproj/free-spoken-digit-dataset-master/data_extraction/model_cnn_2.h5')  

def convert_to_spectrogram(raw_data, sr):
    spect = librosa.feature.melspectrogram(y=raw_data, sr=sr, n_mels=64) 
    mel_spect = librosa.power_to_db(spect, ref=np.max)
    return mel_spect

st.title("Audio Classification with Spectrogram")

uploaded_file = st.file_uploader("Upload a WAV file", type="wav")

if uploaded_file is not None:
    raw_data, sr = librosa.load(uploaded_file, sr=None)

    st.audio(uploaded_file, format='audio/wav')

    st.subheader("Spectrogram")
    spectrogram = convert_to_spectrogram(raw_data, sr)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel Spectrogram")
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    st.image(buf)

    spectrogram = np.expand_dims(spectrogram, axis=0)  
    spectrogram = np.expand_dims(spectrogram, axis=-1) 

    
    prediction = np.argmax(model.predict(spectrogram), axis=1)
    st.subheader("Predicted Class")
    st.write(f"The model predicts this audio belongs to class: {prediction[0]}")

