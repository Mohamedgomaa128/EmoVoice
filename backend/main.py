from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import uvicorn

app = FastAPI(title="EmoVoice")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === استخدام المودل بتاعك ===
model = load_model("models/best_model.h5")   # غير الاسم لو مختلف

EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
EMOJI = ["😠", "🤢", "😨", "😊", "😐", "😢"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load audio
    audio, sr = librosa.load(file.file, sr=22050)
    
    # Preprocessing (نفس الكود بتاعك)
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
    
    if mel_spec.shape[1] >= n_frames:
        mel_spec = mel_spec[:, :n_frames]
    else:
        pad_width = n_frames - mel_spec.shape[1]
        mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), 'constant')

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_input = mel_spec_db.T[:, :, np.newaxis] / -80.0
    mel_spec_input = np.expand_dims(mel_spec_input, axis=0)

    # Prediction
    pred = model.predict(mel_spec_input, verbose=0)
    idx = np.argmax(pred[0])
    confidence = float(pred[0][idx])

    probabilities = {EMOTIONS[i]: round(float(pred[0][i]) * 100, 2) for i in range(6)}

    return {
        "emotion": EMOTIONS[idx],
        "emoji": EMOJI[idx],
        "confidence": round(confidence * 100, 2),
        "probabilities": probabilities
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)