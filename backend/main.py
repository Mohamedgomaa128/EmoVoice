from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import uvicorn
import io

app = FastAPI(
    title="EmoVoice API",
    description="Speech Emotion Recognition using 2D-CNN trained on CREMA-D dataset",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model ──────────────────────────────────────────────────────────────────────
model = load_model("models/best_model.h5")

EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
EMOTION_NAMES = {
    "ANG": "Angry",
    "DIS": "Disgust",
    "FEA": "Fear",
    "HAP": "Happy",
    "NEU": "Neutral",
    "SAD": "Sad",
}
EMOJI = {
    "ANG": "😠",
    "DIS": "🤢",
    "FEA": "😨",
    "HAP": "😊",
    "NEU": "😐",
    "SAD": "😢",
}

# ── Audio preprocessing — mirrors the notebook exactly ──────────────────────────
SR = 22050
ROW_LEN = 56000      # 56 000 samples ≈ 2.54 s at 22 050 Hz
N_MELS = 256
N_FFT = 4096
N_FRAMES = 60


def preprocess_audio(audio_bytes: bytes) -> np.ndarray:
    """Load raw audio bytes, build Mel-spectrogram, return model-ready tensor."""
    audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=SR, mono=True)

    # Pad or truncate to fixed length
    if len(audio) >= ROW_LEN:
        audio = audio[:ROW_LEN]
    else:
        audio = np.pad(audio, (0, ROW_LEN - len(audio)), mode="constant")

    # Dynamic hop_length to always get exactly N_FRAMES columns
    audio_dur = len(audio) / SR
    frame_step = audio_dur / N_FRAMES
    hop_length = int(round(frame_step * SR))

    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=hop_length
    )

    if mel_spec.shape[1] >= N_FRAMES:
        mel_spec = mel_spec[:, :N_FRAMES]
    else:
        pad_width = N_FRAMES - mel_spec.shape[1]
        mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode="constant")

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Shape: (N_FRAMES, N_MELS, 1) — normalised to [0, 1]
    tensor = mel_spec_db.T[:, :, np.newaxis] / -80.0
    return np.expand_dims(tensor, axis=0)  # (1, 60, 256, 1)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "EmoVoice API is running"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    # Validate content type
    if not (file.content_type or "").startswith("audio/"):
        raise HTTPException(status_code=400, detail="Only audio files are accepted.")

    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        tensor = preprocess_audio(audio_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Audio processing failed: {exc}")

    pred = model.predict(tensor, verbose=0)[0]   # shape: (6,)
    idx = int(np.argmax(pred))
    code = EMOTIONS[idx]

    probabilities = {
        EMOTIONS[i]: round(float(pred[i]) * 100, 2) for i in range(len(EMOTIONS))
    }

    return {
        "emotion": code,
        "emotion_name": EMOTION_NAMES[code],
        "emoji": EMOJI[code],
        "confidence": round(float(pred[idx]) * 100, 2),
        "probabilities": probabilities,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
