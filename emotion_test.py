import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_PATH = "D:/SER MiniProj/wav2vec2_model/"
TARGET_SAMPLE_RATE = 16000  # Model requires 16kHz audio

# Load feature extractor and model
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH)

print("Feature extractor and model loaded successfully!")

# Load an audio file
audio_file = "D:/SER MiniProj/temp_audio.wav"
waveform, sample_rate = torchaudio.load(audio_file)

# Convert to mono if needed
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

# Resample if the sample rate is not 16kHz
if sample_rate != TARGET_SAMPLE_RATE:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
    waveform = resampler(waveform)
    sample_rate = TARGET_SAMPLE_RATE  # Update sample rate

# Process the audio for the model
inputs = feature_extractor(waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt")

# Perform inference
with torch.no_grad():
    logits = model(**inputs).logits

# Get the predicted emotion
predicted_label = torch.argmax(logits, dim=-1).item()

# Print the output
print(f"Predicted Emotion Class: {predicted_label}")
