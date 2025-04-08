import streamlit as st
import torch
import torchaudio
import whisper
from transformers import pipeline, AutoFeatureExtractor, AutoModelForAudioClassification
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import altair as alt

# Constants
MODEL_PATH = "D:/SER MiniProj/wav2vec2_model/"
TARGET_SAMPLE_RATE = 16000  # Required sample rate for Wav2Vec2
AUDIO_SAVE_PATH = "temp_audio.wav"

emotion_labels = {
    0: "Neutral",
    1: "Calm",
    2: "Happy",
    3: "Sad",
    4: "Angry",
    5: "Fearful",
    6: "Disgust",
    7: "Surprised"
}

# Load models with caching
@st.cache_resource
def load_models():
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
    ser_model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH)

    whisper_model = whisper.load_model("base")
    summarizer = pipeline("summarization", model="t5-base", framework="pt")

    return feature_extractor, ser_model, whisper_model, summarizer

feature_extractor, ser_model, whisper_model, summarizer = load_models()

# UI Layout
st.set_page_config(page_title="Speech Analysis App", layout="wide")
st.title("Speech Emotion Recognition & Summarization")
st.markdown("Upload an audio file to analyze emotions, transcribe speech, and get a concise summary.")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "ogg"])

if uploaded_file:
    with open(AUDIO_SAVE_PATH, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(AUDIO_SAVE_PATH, format="audio/wav")

    waveform, sample_rate = torchaudio.load(AUDIO_SAVE_PATH)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)

    y = waveform.squeeze().numpy()

    # Audio Visualization
    st.subheader("Audio Visualizations")

    # Waveform
    st.markdown("**Waveform**")
    fig_wave, ax_wave = plt.subplots(figsize=(8, 1.8))  # Reduced vertical height
    ax_wave.plot(y, linewidth=0.8)
    ax_wave.set_xlabel("Samples")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_title("Waveform", fontsize=10)
    ax_wave.tick_params(labelsize=8)
    fig_wave.tight_layout(pad=0.3)
    st.pyplot(fig_wave)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Spectrogram**")
        fig, ax = plt.subplots(figsize=(6, 3))
        D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
        img = librosa.display.specshow(D, sr=TARGET_SAMPLE_RATE, x_axis='time', y_axis='log', ax=ax)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        fig.tight_layout(pad=0.5)
        st.pyplot(fig)

    with col2:
        st.markdown("**MFCCs**")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        mfccs = librosa.feature.mfcc(y=y, sr=TARGET_SAMPLE_RATE, n_mfcc=13)
        img2 = librosa.display.specshow(mfccs, x_axis='time', ax=ax2)
        fig2.colorbar(img2, ax=ax2)
        fig2.tight_layout(pad=0.5)
        st.pyplot(fig2)

    # Emotion Prediction
    st.subheader("Emotion Recognition")
    inputs = feature_extractor(y, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        logits = ser_model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    predicted_emotion = emotion_labels[predicted_class]
    st.success(f"Predicted Emotion: {predicted_emotion}")

    # Transcription & Summarization
    st.subheader("Speech Transcription & Summary")
    transcription = whisper_model.transcribe(AUDIO_SAVE_PATH)["text"]
    st.info(f"Transcription: {transcription}")

    summary = summarizer(transcription, max_length=50, min_length=10, do_sample=False)[0]["summary_text"]
    st.success(f"Summary: {summary}")

    # Playback speed
    st.subheader("Playback Options")
    speed = st.select_slider("Playback Speed", options=[0.5, 0.75, 1.0, 1.25, 1.5], value=1.0)
    st.markdown(f"Playback speed set to {speed}x (you can use external player to preview adjusted audio)")

    # Audio Info
    st.markdown("**Audio Metadata**")
    st.write(f"Duration: {round(len(y) / TARGET_SAMPLE_RATE, 2)} seconds")
    st.write(f"Sample Rate: {TARGET_SAMPLE_RATE} Hz")


# # Streamlit UI
# st.title("🎤 Speech Analysis: Emotion & Summarization")

# # Upload audio file
# uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

# if uploaded_file is not None:
#     # Save the uploaded file
#     with open(AUDIO_SAVE_PATH, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     # Display the audio
#     st.audio(AUDIO_SAVE_PATH, format="audio/wav")

#     # **Speech Emotion Recognition**
#     st.subheader("Speech Emotion Recognition")
#     waveform, sample_rate = torchaudio.load(AUDIO_SAVE_PATH)

#     # Convert stereo to mono
#     if waveform.shape[0] > 1:
#         waveform = torch.mean(waveform, dim=0, keepdim=True)

#     # Resample if needed
#     if sample_rate != TARGET_SAMPLE_RATE:
#         resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
#         waveform = resampler(waveform)

#     # Extract features
#     inputs = feature_extractor(waveform.squeeze(0), sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")

#     # Get emotion prediction
#     with torch.no_grad():
#         logits = ser_model(**inputs).logits

#     predicted_class = torch.argmax(logits, dim=-1).item()
#     emotion = emotion_labels.get(predicted_class, "Unknown")
#     st.success(f"Predicted Emotion: {emotion} ({predicted_class})")

#     # **Speech Summarization**
#     st.subheader(" Speech Summarization")
#     transcription = whisper_model.transcribe(AUDIO_SAVE_PATH)["text"]
#     st.info(f" Transcription: {transcription}")

#     # Generate summary
#     summary = summarizer(transcription, max_length=50, min_length=10, do_sample=False)[0]["summary_text"]
#     st.success(f"The Summary: {summary}")
