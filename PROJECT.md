# EmoVoice — Project Overview & Business Logic

## What Is EmoVoice?

EmoVoice is a full-stack web application that detects human emotions from voice recordings using deep learning. A user records their voice or uploads an audio file; the system analyses the acoustic patterns and returns one of six emotions (Angry, Disgust, Fear, Happy, Neutral, Sad) along with a confidence score and a breakdown of all emotion probabilities.

---

## Problem Statement

Emotion recognition from voice has broad real-world applicability:

| Domain | Use Case |
|---|---|
| Mental Health | Detect distress signals in therapy/call-centre sessions |
| Customer Service | Flag frustrated callers in real time for escalation |
| Education | Monitor student engagement/confusion during e-learning |
| HR & Interviews | Supplement structured interview analysis |
| Gaming / VR | Adaptive NPC responses based on player emotion |
| Accessibility | Assist non-verbal users to express emotional state |

---

## Architecture

```
┌─────────────────────────────┐        HTTP (FormData)        ┌──────────────────────────────────┐
│   Browser (React + Vite)    │  ─────────────────────────►  │  FastAPI Server (Python)          │
│                             │                               │                                  │
│  • Record via MediaRecorder │  ◄─────────────────────────  │  1. Librosa — load & resample     │
│  • Upload audio file        │        JSON response          │  2. Mel-Spectrogram (256×60)      │
│  • WaveSurfer waveform      │                               │  3. 2D-CNN inference              │
│  • Confidence bar chart     │                               │  4. Softmax → 6 probabilities     │
└─────────────────────────────┘                               └──────────────────────────────────┘
```

---

## ML Pipeline

### Dataset — CREMA-D
- **7,442** audio clips from **91 actors** (diverse age, gender, ethnicity)
- 6 emotion classes: ANG · DIS · FEA · HAP · NEU · SAD
- Each clip is labelled with both the emotion and intensity level

### Feature Engineering
Raw audio → fixed-length signal (56 000 samples @ 22 050 Hz ≈ 2.5 s) →
**Mel-Spectrogram** (256 mel bins × 60 time frames) →
Log-power (dB) normalised to [0, 1]

Why Mel-Spectrogram?
- Mimics human auditory perception (logarithmic frequency scale)
- Preserves both spectral and temporal structure
- Outperforms raw waveform features and hand-crafted MFCC features for this task

### Model — 2D-CNN
```
Input: (60, 256, 1)
  Conv2D(32, 3×3, relu) → MaxPool(2×2)
  Conv2D(64, 3×3, relu) → MaxPool(2×2)
  Conv2D(128,3×3, relu) → MaxPool(2×2)
  Conv2D(256,3×3, relu) → MaxPool(2×2)
  Flatten → Dense(256, relu) → Dense(6, softmax)
```
- Treats the spectrogram as an image — learns local spectro-temporal patterns
- Best validation accuracy ≈ **50%** after 9 epochs (baseline; see improvements below)
- Saved as `backend/models/best_model.h5` via `ModelCheckpoint(monitor='val_loss')`

### Preprocessing Alignment
The backend (`main.py`) uses the **exact same** preprocessing as the training notebook:
- Same `ROW_LEN`, `N_MELS`, `N_FFT`, `N_FRAMES`, normalisation factor (`/ -80.0`)
- Dynamic `hop_length` derived from audio duration to guarantee exactly 60 frames

---

## Codebase Structure

```
EmoVoice/
├── backend/
│   ├── main.py             ← FastAPI server + /predict endpoint
│   ├── train_model.py      ← Standalone training script (run offline)
│   ├── requirements.txt    ← Python dependencies
│   └── models/
│       └── best_model.h5   ← Trained weights (not committed to git)
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx         ← Main React UI component
│   │   ├── main.tsx        ← Vite entry point
│   │   └── index.css       ← Tailwind base styles
│   ├── package.json
│   └── vite.config.ts
│
├── files/                  ← Legacy Flask prototype (kept for reference)
│   ├── app.py
│   └── index.html
│
├── Speech emotion recognition/   ← Research notebook & earlier SER project
│   ├── app.py
│   └── Readme.md
│
├── README.md
└── PROJECT.md              ← (this file)
```

---

## Running the Project

### Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
# Server starts at http://localhost:8000
# Docs at  http://localhost:8000/docs
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# App opens at http://localhost:5173
```

### Training (offline only — do NOT run while server is running)
```bash
cd backend
python train_model.py --data_path /path/to/CREMA-D --epochs 30
```

---

## Improvement Roadmap

Below are prioritised enhancements ordered by impact vs. effort.

### 1. Model Accuracy (High Impact)
| Idea | Expected Gain |
|---|---|
| Data augmentation (time-stretch, pitch-shift, noise) | +5–10% acc |
| Transfer learning (VGGish, wav2vec 2.0) | +15–20% acc |
| Add Dropout / BatchNorm layers | Reduce overfitting |
| Increase epochs + LR scheduling | Better convergence |
| Stratified k-fold cross-validation | More reliable evaluation |
| Ensemble 1D-CNN + 2D-CNN | +3–5% acc |

### 2. Backend (Medium Impact)
| Idea | Effort |
|---|---|
| File size & duration validation | Low |
| Async model inference (background task) | Medium |
| Redis caching for repeated audio | Medium |
| Docker containerisation | Low |
| Unit tests for preprocessing pipeline | Low |

### 3. Frontend (Medium Impact)
| Idea | Effort |
|---|---|
| Real-time emotion meter during recording (Web Audio API) | Medium |
| History of past analyses (localStorage) | Low |
| Dark/light mode toggle | Low |
| Internationalisation (Arabic + English toggle) | Low |
| Share result as image card | Medium |

### 4. Deployment (Low Effort, High Portfolio Value)
| Platform | Notes |
|---|---|
| Render / Railway | Free tier, easy FastAPI deploy |
| Vercel | Frontend static deployment |
| HuggingFace Spaces | Great for ML demos + model hosting |
| Docker Compose | One-command local setup |

---

## Tech Stack Summary

| Layer | Technology | Why |
|---|---|---|
| UI Framework | React 18 + TypeScript | Type safety, component model |
| Build Tool | Vite | Fast HMR, lean bundles |
| Styling | Tailwind CSS | Utility-first, consistent dark theme |
| Audio Playback | WaveSurfer.js | Interactive waveform visualisation |
| HTTP Client | Axios | Interceptors, typed responses |
| API Server | FastAPI | Auto-docs, async, high performance |
| ML Framework | TensorFlow / Keras | Model training & inference |
| Audio Processing | Librosa | Industry-standard audio feature extraction |
| Model Format | Keras HDF5 (.h5) | Portable, single-file weights |
