import whisper
from transformers import pipeline

AUDIO_FILE = "D:/SER MiniProj/temp_audio.wav"
# Load Whisper model for transcription
whisper_model = whisper.load_model("base")  # You can use "small", "medium", or "large" for better accuracy

# Transcribe the audio
transcription = whisper_model.transcribe(AUDIO_FILE)["text"]
print(f"📝 Transcribed Text: {transcription}")

# Load summarization model
summarizer = pipeline("summarization", model="t5-base", framework="pt")

# Generate summary
summary = summarizer(transcription, max_length=50, min_length=10, do_sample=False)[0]["summary_text"]
print(f"📌 Summary: {summary}")