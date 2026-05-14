from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import uvicorn
import os

app = FastAPI(title="EmoVoice - Speech Emotion Recognition", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
model = load_model("models/best_model_spectrogram.h5")

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
EMOJI = ["😠", "🤢", "😨", "😊", "😐", "😢"]

@app.get("/")
async def root():
    return {"message": "EmoVoice API is running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(('.wav', '.mp3', '.m4a')):
        raise HTTPException(400, detail="Only audio files are supported")

    try:
        # Read audio
        audio, sr = librosa.load(file.file, sr=22050)
        
        # Preprocessing (matching your notebook)
        row_len = 56000
        if len(audio) >= row_len:
            audio = audio[:row_len]
        else:
            audio = np.pad(audio, (0, row_len - len(audio)))

        # Mel Spectrogram
        n_mels = 256
        n_fft = 4096
        hop_length = 1024
        n_frames = 60

        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        mel_spec = np.pad(mel_spec, ((0, 0), (0, max(0, n_frames - mel_spec.shape[1]))), mode='constant')
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = mel_spec_db.T[:n_frames, :, np.newaxis] / -80.0
        mel_spec_db = np.expand_dims(mel_spec_db, axis=0)

        # Predict
        prediction = model.predict(mel_spec_db, verbose=0)
        emotion_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][emotion_idx])

        probabilities = {EMOTIONS[i]: round(float(prediction[0][i]) * 100, 2) for i in range(6)}

        return JSONResponse({
            "emotion": EMOTIONS[emotion_idx],
            "emoji": EMOJI[emotion_idx],
            "confidence": round(confidence * 100, 2),
            "probabilities": probabilities
        })

    except Exception as e:
        raise HTTPException(500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)