"""
train_model.py — EmoVoice Model Training Script
================================================
This script trains the 2D-CNN model on the CREMA-D dataset and saves the best
checkpoint to models/best_model.h5.

DO NOT run this in the web server.  Run it once offline / on Colab to produce
the .h5 file, then place it at backend/models/best_model.h5.

Usage (local, after installing requirements):
    python train_model.py --data_path /path/to/CREMA-D

Usage (Google Colab):
    Mount Drive, adjust DATA_PATH, then run all cells.
"""

import argparse
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

# ── Hyper-parameters (must match backend/main.py) ─────────────────────────────
SR = 22050
ROW_LEN = 56000          # fixed signal length in samples
N_MELS = 256
N_FFT = 4096
N_FRAMES = 60

EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]

# ── Feature extraction ─────────────────────────────────────────────────────────

def load_crema(data_path: str):
    """Load CREMA-D wav files, return Mel-spectrogram tensors and labels."""
    spect_data = []
    labels = []

    for file_name in sorted(os.listdir(data_path)):
        if not file_name.endswith(".wav"):
            continue

        file_path = os.path.join(data_path, file_name)
        # CREMA-D filename format: <ActorID>_<SentenceID>_<Emotion>_<Level>.wav
        emotion_code = file_name.split("_")[2]
        if emotion_code not in EMOTIONS:
            continue

        audio, _ = librosa.load(file_path, sr=SR, mono=True)

        # Pad / truncate
        if len(audio) >= ROW_LEN:
            audio = audio[:ROW_LEN]
        else:
            audio = np.pad(audio, (0, ROW_LEN - len(audio)), mode="constant")

        spect_data.append(_mel_spectrogram(audio))
        labels.append(emotion_code)

    return np.array(spect_data), np.array(labels)


def _mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Compute Mel-spectrogram tensor of shape (N_FRAMES, N_MELS, 1)."""
    audio_dur = len(audio) / SR
    frame_step = audio_dur / N_FRAMES
    hop_length = int(round(frame_step * SR))

    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=hop_length
    )

    if mel_spec.shape[1] >= N_FRAMES:
        mel_spec = mel_spec[:, :N_FRAMES]
    else:
        mel_spec = np.pad(mel_spec, ((0, 0), (0, N_FRAMES - mel_spec.shape[1])), "constant")

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db.T[:, :, np.newaxis] / -80.0   # shape: (60, 256, 1), range [0, 1]


# ── Label encoding ─────────────────────────────────────────────────────────────

def encode_labels(labels: np.ndarray) -> np.ndarray:
    return np.array([EMOTIONS.index(lbl) for lbl in labels])


# ── Model architecture ─────────────────────────────────────────────────────────

def build_model(input_shape=(N_FRAMES, N_MELS, 1), n_classes=6) -> models.Sequential:
    model = models.Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0006),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    return model


# ── Training ───────────────────────────────────────────────────────────────────

def train(data_path: str, output_path: str = "models/best_model.h5", epochs: int = 20):
    print(f"[EmoVoice] Loading CREMA-D from: {data_path}")
    X, y_str = load_crema(data_path)
    y = encode_labels(y_str)
    print(f"[EmoVoice] Loaded {len(X)} samples, shape {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y_str
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.05, random_state=42
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    checkpoint = ModelCheckpoint(output_path, monitor="val_loss", save_best_only=True, verbose=1)

    model = build_model()
    model.summary()

    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint],
    )

    # Evaluate on test set
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[EmoVoice] Test accuracy: {acc:.4f}  |  Test loss: {loss:.4f}")
    print(f"[EmoVoice] Best model saved to: {output_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EmoVoice 2D-CNN on CREMA-D")
    parser.add_argument("--data_path", required=True, help="Path to CREMA-D wav files directory")
    parser.add_argument("--output", default="models/best_model.h5", help="Where to save best_model.h5")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    args = parser.parse_args()

    train(args.data_path, args.output, args.epochs)
