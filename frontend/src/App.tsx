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
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  const initWaveSurfer = (audioUrl: string) => {
    if (wavesurfer.current) wavesurfer.current.destroy();
    wavesurfer.current = WaveSurfer.create({
      container: waveformRef.current!,
      waveColor: '#818cf8',
      progressColor: '#6366f1',
      cursorColor: '#a5b4fc',
      height: 110,
      barWidth: 3,
    });
    wavesurfer.current.load(audioUrl);
  };

  const handleFileUpload = (selectedFile: File) => {
    setFile(selectedFile);
    setPrediction(null);
    initWaveSurfer(URL.createObjectURL(selectedFile));
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => audioChunksRef.current.push(e.data);
      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const audioFile = new File([blob], "recording.wav", { type: "audio/wav" });
        setFile(audioFile);
        initWaveSurfer(URL.createObjectURL(blob));
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (err) {
      alert("لا يمكن الوصول للميكروفون");
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    streamRef.current?.getTracks().forEach(track => track.stop());
    setIsRecording(false);
  };

  const predictEmotion = async () => {
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
            className={`flex items-center gap-3 px-8 py-4 rounded-2xl font-semibold transition ${isRecording ? 'bg-red-600 hover:bg-red-700' : 'bg-indigo-600 hover:bg-indigo-700'}`}
          >
            {isRecording ? <Square size={24} /> : <Mic size={24} />}
            {isRecording ? 'إيقاف التسجيل' : 'تسجيل صوت'}
          </button>

          <label className="flex items-center gap-3 px-8 py-4 bg-slate-800 hover:bg-slate-700 rounded-2xl cursor-pointer">
            <Upload size={24} /> رفع ملف صوتي
            <input type="file" accept="audio/*" className="hidden" onChange={(e) => e.target.files && handleFileUpload(e.target.files[0])} />
          </label>
        </div>

        {(file || isRecording) && (
          <div className="bg-slate-900 rounded-3xl p-8 mb-8">
            <div ref={waveformRef} className="mb-6" />
            <div className="flex gap-4 justify-center">
              <button onClick={() => wavesurfer.current?.playPause()} className="px-6 py-3 bg-slate-800 rounded-xl">
                {isPlaying ? <Pause /> : <Play />}
              </button>
              <button onClick={predictEmotion} disabled={loading} className="px-10 py-3 bg-indigo-600 rounded-xl font-semibold">
                {loading ? 'جاري التحليل...' : '🔍 تحليل العاطفة'}
              </button>
            </div>
          </div>
        )}

        {prediction && (
          <div className="bg-slate-900 rounded-3xl p-10">
            <div className="text-center">
              <div className="text-9xl mb-6">{prediction.emoji}</div>
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