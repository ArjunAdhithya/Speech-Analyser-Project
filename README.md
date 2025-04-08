# Speech Analyser Project 

This project presents a unified system for **Speech Emotion Recognition**, **Speech Transcription**, and **Speech Summarization**, designed to extract both emotional and semantic meaning from spoken audio input. Built with deep learning models and deployed through a user-friendly web interface, the tool aims to assist in real-world applications like mental health tracking, classroom feedback evaluation, and customer interaction analysis.

---

## Objective

The primary goal of this project is to automate the process of understanding both *what* is being said in speech and *how* it is being said, addressing the need for emotional and contextual insight in spoken communication.

### Key Use Cases:
- Identifying emotional tones in therapy or support calls
- Summarizing classroom lectures or student feedback
- Evaluating customer service interactions for sentiment and clarity
- Supporting emergency response by analyzing stress or urgency in calls

---

##  Features

-  **Speech Emotion Detection** using `Wav2Vec2`
-  **Speech Transcription** using `OpenAI Whisper`
-  **Summarization** using a transformer-based language model
-  **Audio Visualizations** including:
  - Waveform
  - Spectrogram
  - MFCCs
-  **Streamlit Web Interface** for interactive, real-time analysis

---

## Technologies Used

- Python  
- Streamlit  
- Transformers (`Wav2Vec2`, `Whisper`)  
- Librosa, Matplotlib, Seaborn (for audio processing and visualization)  
- Hugging Face Transformers  
- Numpy, Scikit-learn

---

## Project Workflow

1. **Input**: Upload an audio file (WAV/MP3)
2. **Emotion Detection**: Classify emotional state from raw audio
3. **Transcription**: Convert speech to text using Whisper
4. **Summarization**: Generate a short summary of the transcribed content
5. **Visualization**: Display audio waveform, spectrogram, and MFCCs
6. **Output**: View emotion result, full transcript, and summary

---

## Directory Structure

```bash
├── app.py                 # Streamlit web app
├── requirements.txt       # Dependencies
├── utils/
│   ├── audio_processing.py
│   ├── emotion_detection.py
│   ├── transcription.py
│   └── summarization.py
├── data/
│   └── sample_audio/
├── models/
│   └── saved_model/
└── README.md
