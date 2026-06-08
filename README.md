# EmoVoice — Speech Emotion Recognition

![React](https://img.shields.io/badge/React-18.3-61DAFB?logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow)
![TypeScript](https://img.shields.io/badge/TypeScript-5.5-3178C6?logo=typescript)

A full-stack web application that classifies human emotions from voice recordings using a **2D Convolutional Neural Network** trained on the [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) dataset.

---

## Features

- 🎤 **Live microphone recording** with real-time timer
- 📁 **Audio file upload** (wav, mp3, ogg, …) with drag-and-drop support
- 📊 **Interactive waveform** visualisation via WaveSurfer.js
- 🧠 **Emotion inference** powered by a 2D-CNN Mel-Spectrogram model
- 📈 **Probability bar chart** for all 6 emotions
- 🌑 Modern dark UI, fully responsive

### Supported Emotions

| Code | Emotion | Emoji |
|------|---------|-------|
| ANG | Angry | 😠 |
| DIS | Disgust | 🤢 |
| FEA | Fear | 😨 |
| HAP | Happy | 😊 |
| NEU | Neutral | 😐 |
| SAD | Sad | 😢 |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18 + TypeScript + Vite + Tailwind CSS |
| Audio UI | WaveSurfer.js |
| HTTP | Axios |
| Backend | FastAPI + Uvicorn |
| ML | TensorFlow 2.15 / Keras |
| Audio processing | Librosa (Mel-Spectrogram) |

---

## Quick Start

### Prerequisites
- Python ≥ 3.10
- Node.js ≥ 18
- `backend/models/best_model.h5` (place your trained model here)

### 1 — Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
```

Server runs at **http://localhost:8000**
Interactive API docs at **http://localhost:8000/docs**

### 2 — Frontend

```bash
cd frontend
npm install
npm run dev
```

App opens at **http://localhost:5173**

---

## API Reference

### `POST /predict`

Accepts a multipart audio file and returns emotion predictions.

**Request**
```
Content-Type: multipart/form-data
Body: file=<audio_file>
```

**Response**
```json
{
  "emotion": "HAP",
  "emotion_name": "Happy",
  "emoji": "😊",
  "confidence": 83.47,
  "probabilities": {
    "ANG": 2.10,
    "DIS": 1.55,
    "FEA": 3.22,
    "HAP": 83.47,
    "NEU": 7.91,
    "SAD": 1.75
  }
}
```

### `GET /health`
Returns `{"status": "healthy", "model_loaded": true}` — useful for monitoring.

---

## Model Details

| Property | Value |
|----------|-------|
| Dataset | CREMA-D (7 442 clips, 91 actors) |
| Input feature | Mel-Spectrogram (256 mel bins × 60 frames) |
| Architecture | 4-block 2D-CNN → Dense(256) → Softmax(6) |
| Training | Adam lr=0.0006, ModelCheckpoint on val_loss |
| Best val accuracy | ~50% (9 epochs baseline) |

To retrain the model offline:
```bash
cd backend
python train_model.py --data_path /path/to/CREMA-D --epochs 30
```

> ⚠️ **Never run training while the API server is running** — it will exhaust compute resources.

---

## Project Structure

```
EmoVoice/
├── backend/
│   ├── main.py           ← FastAPI app
│   ├── train_model.py    ← Offline training script
│   ├── requirements.txt
│   └── models/
│       └── best_model.h5
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   └── main.tsx
│   └── package.json
├── PROJECT.md            ← Architecture, business logic & roadmap
└── README.md
```

See **[PROJECT.md](./PROJECT.md)** for architecture diagrams, ML pipeline details, and the full improvement roadmap.

---

## License

MIT
