# EmoVoice - Speech Emotion Recognition

![React](https://img.shields.io/badge/React-18.3-61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00)
![TypeScript](https://img.shields.io/badge/TypeScript-5.5-3178C6)

**تطبيق ويب احترافي لتحليل العواطف من الصوت باستخدام الذكاء الاصطناعي**

## ✨ المميزات

- 🎤 **تسجيل صوت حي** مباشرة من الميكروفون (Real-time Recording)
- 📁 رفع ملفات صوتية (wav, mp3, ...)
- 📊 عرض **Waveform** تفاعلي أثناء التشغيل والتسجيل
- 🧠 تحليل فوري باستخدام **Convolutional Neural Network**
- 📈 عرض نسب الثقة لكل عاطفة بشكل بصري جميل
- 🎨 واجهة حديثة Dark Mode + متجاوبة مع جميع الأجهزة

## العواطف المدعومة
**Angry** 😠 | **Disgust** 🤢 | **Fear** 😨 | **Happy** 😊 | **Neutral** 😐 | **Sad** 😢

---

## 🛠 التقنيات المستخدمة

- **Frontend**: React + TypeScript + Vite + Tailwind CSS
- **Backend**: FastAPI + TensorFlow
- **Audio Processing**: Librosa (Mel Spectrogram)
- **Visualization**: wavesurfer.js
- **HTTP Client**: Axios

---

## كيفية التشغيل (Local)

### 1. Backend

```bash
cd backend

# تثبيت المتطلبات
pip install -r requirements.txt

# تشغيل السيرفر
python main.py