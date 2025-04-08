import whisper
model = whisper.load_model("small")
result = model.transcribe("temp_audio.wav")
print("Whisper Output:", result["text"])