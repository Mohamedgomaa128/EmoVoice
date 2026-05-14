import { useState, useRef } from 'react';
import WaveSurfer from 'wavesurfer.js';
import axios from 'axios';
import { Upload, Play, Pause, Mic, Square } from 'lucide-react';

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isRecording, setIsRecording] = useState(false);

  const wavesurfer = useRef<WaveSurfer | null>(null);
  const waveformRef = useRef<HTMLDivElement>(null);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  const initWaveSurfer = (url: string) => {
    if (wavesurfer.current) wavesurfer.current.destroy();
    wavesurfer.current = WaveSurfer.create({
      container: waveformRef.current!,
      waveColor: '#818cf8',
      progressColor: '#6366f1',
      cursorColor: '#a5b4fc',
      height: 110,
      barWidth: 3,
    });
    wavesurfer.current.load(url);
  };

  const handleFile = (selectedFile: File) => {
    setFile(selectedFile);
    setPrediction(null);
    initWaveSurfer(URL.createObjectURL(selectedFile));
  };

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    streamRef.current = stream;
    mediaRecorder.current = new MediaRecorder(stream);
    audioChunks.current = [];

    mediaRecorder.current.ondataavailable = e => audioChunks.current.push(e.data);
    mediaRecorder.current.onstop = () => {
      const blob = new Blob(audioChunks.current, { type: 'audio/wav' });
      const newFile = new File([blob], "recording.wav", { type: "audio/wav" });
      setFile(newFile);
      initWaveSurfer(URL.createObjectURL(blob));
    };

    mediaRecorder.current.start();
    setIsRecording(true);
  };

  const stopRecording = () => {
    mediaRecorder.current?.stop();
    streamRef.current?.getTracks().forEach(track => track.stop());
    setIsRecording(false);
  };

  const predict = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await axios.post('http://localhost:8000/predict', formData);
      setPrediction(res.data);
    } catch (err) {
      alert("خطأ في الاتصال بالسيرفر");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-950 text-white p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-6xl font-bold text-center mb-3">EmoVoice</h1>
        <p className="text-center text-slate-400 mb-12">Speech Emotion Recognition</p>

        <div className="flex gap-4 justify-center mb-8">
          <button
            onClick={isRecording ? stopRecording : startRecording}
            className={`flex items-center gap-3 px-8 py-4 rounded-2xl font-semibold ${isRecording ? 'bg-red-600' : 'bg-indigo-600'}`}
          >
            {isRecording ? <Square /> : <Mic />} 
            {isRecording ? 'إيقاف التسجيل' : 'تسجيل صوت'}
          </button>

          <label className="flex items-center gap-3 px-8 py-4 bg-slate-800 hover:bg-slate-700 rounded-2xl cursor-pointer">
            <Upload /> رفع ملف
            <input type="file" accept="audio/*" className="hidden" onChange={e => e.target.files && handleFile(e.target.files[0])} />
          </label>
        </div>

        {(file || isRecording) && (
          <div className="bg-slate-900 rounded-3xl p-8 mb-8">
            <div ref={waveformRef} className="mb-6" />
            <div className="flex gap-4 justify-center">
              <button onClick={() => wavesurfer.current?.playPause()} className="px-6 py-3 bg-slate-800 rounded-xl">
                {isPlaying ? <Pause /> : <Play />} 
              </button>
              <button onClick={predict} disabled={loading} className="px-10 py-3 bg-indigo-600 rounded-xl font-semibold">
                {loading ? 'جاري التحليل...' : '🔍 تحليل العاطفة'}
              </button>
            </div>
          </div>
        )}

        {prediction && (
          <div className="bg-slate-900 rounded-3xl p-10">
            <div className="text-center mb-8">
              <div className="text-8xl mb-4">{prediction.emoji}</div>
              <h2 className="text-5xl font-bold">{prediction.emotion}</h2>
              <p className="text-3xl text-indigo-400 mt-2">{prediction.confidence}% ثقة</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;