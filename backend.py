from fastapi import FastAPI, File, UploadFile
import uvicorn
import openai
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
import whisper
import os

app = FastAPI()

# Load Whisper model for transcription
whisper_model = whisper.load_model("small")

# Load speech emotion recognition model
ser_model_name = "superb/wav2vec2-base-superb-er"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(ser_model_name)
ser_model = AutoModelForAudioClassification.from_pretrained(ser_model_name)

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure you set this in the terminal before running

@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    try:
        print(f"✅ File received: {file.filename}")

        # Save audio
        audio_path = "temp_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(await file.read())
        print("✅ Audio saved successfully!")

        # 🟢 TEST 1: Check if the file is corrupted
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            print(f"✅ Audio loaded! Shape: {waveform.shape}, Sample Rate: {sample_rate}")
        except Exception as e:
            return {"error": f"❌ Audio loading failed: {e}"}

        # 🟢 TEST 2: Whisper Transcription
        try:
            transcription = whisper_model.transcribe(audio_path)["text"]
            print(f"✅ Whisper Transcription: {transcription}")
        except Exception as e:
            return {"error": f"❌ Whisper failed: {e}"}

        # 🟢 TEST 3: Emotion Recognition
        try:
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sample_rate != 16000:
                resampler = T.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = ser_model(**inputs).logits
            predicted_class = torch.argmax(logits, dim=-1).item()
            emotions = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
            emotion_detected = emotions[predicted_class] if predicted_class < len(emotions) else "unknown"
            print(f"✅ Emotion Detected: {emotion_detected}")
        except Exception as e:
            return {"error": f"❌ Emotion recognition failed: {e}"}

        # 🟢 TEST 4: OpenAI API Summarization
        try:
            summary_response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "Summarize the following text."},
                    {"role": "user", "content": transcription}
                ]
            )
            summary = summary_response["choices"][0]["message"]["content"]
            print(f"✅ OpenAI Summary: {summary}")
        except Exception as e:
            return {"error": f"❌ OpenAI Summarization failed: {e}"}

        return {
            "transcription": transcription,
            "emotion": emotion_detected,
            "summary": summary
        }

    except Exception as e:
        print(f"❌ Error in process_audio: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
