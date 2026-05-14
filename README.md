# EmoVoice - Speech Emotion Recognition

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![React](https://img.shields.io/badge/React-18.3-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)

**تطبيق ويب يتعرف على العواطف من الصوت باستخدام الذكاء الاصطناعي**

## ✨ المميزات

- 🎤 رفع الملفات بالـ Drag & Drop
- 📊 عرض Waveform تفاعلي (wavesurfer.js)
- 🧠 نموذج CNN مدرب على **CREMA-D Dataset**
- 📈 عرض نسب الثقة لكل عاطفة
- 🎨 واجهة حديثة Dark Mode
- ⚡ سريع ومتجاوب

## 🛠 التقنيات المستخدمة

- **Frontend**: React + TypeScript + Vite + Tailwind CSS
- **Backend**: FastAPI + TensorFlow
- **Visualization**: wavesurfer.js
- **Dataset**: CREMA-D (7442 عينة)

## لقطات من التطبيق

*(هتحط صور هنا بعد ما تعمل run)*

![Demo](screenshots/demo.png)

## كيفية التشغيل محلياً

### 1. Backend
```bash
cd backend
pip install -r requirements.txt

# حط الموديل في مجلد models/
python main.py