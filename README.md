# EmoVoice — Speech Emotion Recognition

![React](https://img.shields.io/badge/React-18.3-61DAFB?logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow)
![TypeScript](https://img.shields.io/badge/TypeScript-5.5-3178C6?logo=typescript)

EmoVoice is a modern web application that classifies human emotions from voice recordings using a **2D Convolutional Neural Network** trained on the [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) dataset. 

The application has been updated with modern AI-agent techniques, a highly polished dark UI, and abstract indicators for emotions to maintain user privacy and visual simplicity.

---

## 🌟 Key Features

- 🎤 **Live Microphone Recording:** Record voice directly from the browser with an interactive duration timer.
- 📁 **Drag-and-Drop Upload:** Easily drag and drop or select local audio files (`.wav`, `.mp3`, `.ogg`, etc.).
- 📊 **Interactive Waveform:** Visualize the audio waveform in real time using `WaveSurfer.js`.
- 🧠 **AI-Powered Emotion Inference:** Instantly detect emotions using the 2D-CNN Mel-Spectrogram model.
- 🖼️ **Abstract Visual Indicators:** Uses modern vector icons with glowing neon effects instead of traditional facial emojis to represent emotions.
- 🌐 **Cloud Integration (Google Colab):** Option to run the deep learning backend on Google Colab to avoid installing heavy local packages.

---

## 📸 Application Screenshots

Here is a preview of the EmoVoice web interface:

### 1. Home Page (Ready State)
The clean, modern dashboard where users can choose to record voice or upload audio files.
![Home Page](./Emo%20Screens/Screenshot%202026-06-09%20093434.png)

### 2. Audio File Loaded (Ready for analysis)
Once a local audio file is uploaded, the interface shows the file information and options to play or analyze it.
![Audio File Loaded](./Emo%20Screens/Screenshot%202026-06-09%20093549.png)

### 3. Active Microphone Recording
Real-time indicator showing that recording is in progress with an active duration timer.
![Microphone Recording](./Emo%20Screens/Screenshot%202026-06-09%20094003.png)

### 4. Waveform Visualization
The recorded audio is processed, generating a beautiful, interactive wave visualization.
![Waveform Visualization](./Emo%20Screens/Screenshot%202026-06-09%20094319.png)

### 5. Emotion Detection Result — Happy
The AI outputs the emotion prediction "Happy" (visualized using a glowing Sun icon) along with its probability breakdown.
![Happy Emotion](./Emo%20Screens/Screenshot%202026-06-09%20094217.png)

### 6. Emotion Detection Result — Sad
The AI outputs the emotion prediction "Sad" (visualized using a glowing Cloud Rain icon) with a custom probability layout.
![Sad Emotion](./Emo%20Screens/Screenshot%202026-06-09%20094343.png)

---

## 🎭 Supported Emotions & Abstract Icons

To respect preferences and create a modern, sleek interface, EmoVoice maps speech emotions to glowing abstract vectors:

| Code | Emotion | Symbol | Visual Metaphor |
|------|---------|--------|-----------------|
| **ANG** | Angry | 🔥 Flame | Red heat / Passionate anger |
| **DIS** | Disgust | 👎 Thumbs Down | Green aversion / Disapproval |
| **FEA** | Fear | ⚠️ Warning | Purple caution / Alert state |
| **HAP** | Happy | ☀️ Sun | Golden brightness / Joy |
| **NEU** | Neutral | ⚖️ Scale | Slate balance / Calmness |
| **SAD** | Sad | 🌧️ Cloud Rain | Blue precipitation / Melancholy |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 18 + TypeScript + Vite + Tailwind CSS |
| **Audio UI** | WaveSurfer.js |
| **HTTP client** | Axios |
| **Backend** | FastAPI + Uvicorn |
| **ML Engine** | TensorFlow 2.15 / Keras |
| **Audio Processing** | Librosa (Mel-Spectrogram translation) |

---

## 🚀 Quick Start

### 1. Frontend Setup
Run the development server locally:
```bash
cd frontend
npm install
npm run dev
```
The interface will be served at `http://localhost:5173`.

### 2. Backend Setup

You can run the backend in one of two ways:

#### Option A: Local Run (Requires local Python environment)
```bash
cd backend
pip install -r requirements.txt
python main.py
```
The API server will listen on `http://localhost:8000`.

#### Option B: Google Colab Run (Recommended for lightweight setup)
If you want to run the model without downloading heavy packages (like TensorFlow and Librosa) to your local machine:
1. Open a new notebook on [Google Colab](https://colab.research.google.com).
2. Upload the `backend/main.py` and `backend/models/best_model.h5` files.
3. Run the following code in a cell to start the server and expose it via **ngrok**:

```python
# 1. Install lightweight requirements
!pip install fastapi uvicorn librosa nest-asyncio pyngrok

import nest_asyncio
from pyngrok import ngrok
import uvicorn
import threading
import time

# 2. Kill existing ngrok tunnels
ngrok.kill()
ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")

# 3. Start FastAPI server in a background thread
from main import app
nest_asyncio.apply()

def start_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

threading.Thread(target=start_fastapi, daemon=True).start()
time.sleep(2)

# 4. Open ngrok tunnel
public_url = ngrok.connect(8000)
print("\n🔥 Backend running on Google Colab!")
print("Tunnel URL:", public_url.public_url)
```

4. Copy the generated `https://xxxx.ngrok-free.dev` URL.
5. In the local frontend file `frontend/src/App.tsx` (around line 132), replace `http://localhost:8000/predict` with your new Colab tunnel URL:
```typescript
const res = await axios.post<Prediction>('https://YOUR_TUNNEL_URL.ngrok-free.dev/predict', formData);
```

---

## 📂 Project Structure

```
EmoVoice/
├── backend/
│   ├── main.py           ← FastAPI app
│   ├── train_model.py    ← Offline model training script
│   ├── requirements.txt
│   └── models/
│       └── best_model.h5
├── frontend/
│   ├── src/
│   │   ├── App.tsx       ← React application code
│   │   └── main.tsx
│   └── package.json
├── Emo Screens/          ← Project screenshots
├── PROJECT.md            ← Architecture documentation
└── README.md             ← This file
```

---

## 📄 License

MIT License.
