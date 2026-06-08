import { useState, useRef, useEffect, useCallback } from 'react';
import WaveSurfer from 'wavesurfer.js';
import axios from 'axios';
import { Upload, Play, Pause, Mic, Square, Brain, AudioLines, RefreshCw } from 'lucide-react';

interface Prediction {
  emotion: string;
  emoji: string;
  confidence: number;
  probabilities: Record<string, number>;
}

const EMOTION_LABELS: Record<string, string> = {
  ANG: 'Angry',
  DIS: 'Disgust',
  FEA: 'Fear',
  HAP: 'Happy',
  NEU: 'Neutral',
  SAD: 'Sad',
};

const EMOTION_COLORS: Record<string, string> = {
  ANG: 'bg-red-500',
  DIS: 'bg-green-700',
  FEA: 'bg-purple-500',
  HAP: 'bg-yellow-400',
  NEU: 'bg-slate-400',
  SAD: 'bg-blue-400',
};

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recordingDuration, setRecordingDuration] = useState(0);

  const wavesurfer = useRef<WaveSurfer | null>(null);
  const waveformRef = useRef<HTMLDivElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const initWaveSurfer = useCallback((audioUrl: string) => {
    if (wavesurfer.current) wavesurfer.current.destroy();
    const ws = WaveSurfer.create({
      container: waveformRef.current!,
      waveColor: '#818cf8',
      progressColor: '#6366f1',
      cursorColor: '#a5b4fc',
      height: 100,
      barWidth: 3,
      barGap: 2,
      barRadius: 3,
    });
    ws.load(audioUrl);
    ws.on('play', () => setIsPlaying(true));
    ws.on('pause', () => setIsPlaying(false));
    ws.on('finish', () => setIsPlaying(false));
    wavesurfer.current = ws;
  }, []);

  const handleFileUpload = (selectedFile: File) => {
    if (!selectedFile.type.startsWith('audio/')) {
      setError('يرجى رفع ملف صوتي فقط');
      return;
    }
    setFile(selectedFile);
    setPrediction(null);
    setError(null);
    initWaveSurfer(URL.createObjectURL(selectedFile));
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];
      setRecordingDuration(0);

      mediaRecorderRef.current.ondataavailable = (e) => audioChunksRef.current.push(e.data);
      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const audioFile = new File([blob], 'recording.wav', { type: 'audio/wav' });
        setFile(audioFile);
        setPrediction(null);
        setError(null);
        initWaveSurfer(URL.createObjectURL(blob));
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);

      timerRef.current = setInterval(() => {
        setRecordingDuration((d) => d + 1);
      }, 1000);
    } catch {
      setError('لا يمكن الوصول للميكروفون — تأكد من منح الإذن');
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    setIsRecording(false);
    if (timerRef.current) clearInterval(timerRef.current);
  };

  const predictEmotion = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await axios.post<Prediction>('http://localhost:8000/predict', formData);
      setPrediction(res.data);
    } catch {
      setError('تعذّر الاتصال بالسيرفر — تأكد أن الـ backend شغّال على المنفذ 8000');
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    if (wavesurfer.current) wavesurfer.current.destroy();
    wavesurfer.current = null;
    setFile(null);
    setPrediction(null);
    setError(null);
    setIsPlaying(false);
  };

  // Drag-and-drop handlers
  const onDragOver = (e: React.DragEvent) => { e.preventDefault(); setIsDragging(true); };
  const onDragLeave = () => setIsDragging(false);
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) handleFileUpload(dropped);
  };

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      wavesurfer.current?.destroy();
    };
  }, []);

  const formatDuration = (s: number) => `${Math.floor(s / 60).toString().padStart(2, '0')}:${(s % 60).toString().padStart(2, '0')}`;

  const sortedProbabilities = prediction
    ? Object.entries(prediction.probabilities).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-indigo-950 to-slate-950 text-white">
      {/* Header */}
      <header className="border-b border-white/5 bg-black/20 backdrop-blur-sm px-8 py-4">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-indigo-600 rounded-xl flex items-center justify-center">
              <AudioLines size={20} />
            </div>
            <span className="text-xl font-bold tracking-tight">EmoVoice</span>
          </div>
          <span className="text-xs text-slate-500 hidden sm:block">Speech Emotion Recognition · CREMA-D · CNN</span>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-8 py-12">
        {/* Hero */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold mb-3 bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
            تحليل المشاعر الصوتية
          </h1>
          <p className="text-slate-400 text-lg">سجّل صوتك أو ارفع ملفاً صوتياً وسنكشف العاطفة الكامنة فيه</p>
        </div>

        {/* Action Buttons */}
        <div
          className={`flex flex-col sm:flex-row gap-4 justify-center mb-8 transition-all ${isDragging ? 'scale-105' : ''}`}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
        >
          <button
            onClick={isRecording ? stopRecording : startRecording}
            className={`flex items-center justify-center gap-3 px-8 py-4 rounded-2xl font-semibold transition-all shadow-lg ${
              isRecording
                ? 'bg-red-600 hover:bg-red-700 shadow-red-900/40 animate-pulse'
                : 'bg-indigo-600 hover:bg-indigo-700 shadow-indigo-900/40'
            }`}
          >
            {isRecording ? <Square size={22} /> : <Mic size={22} />}
            {isRecording ? (
              <span>إيقاف التسجيل {formatDuration(recordingDuration)}</span>
            ) : (
              'تسجيل صوت'
            )}
          </button>

          <label
            className={`flex items-center justify-center gap-3 px-8 py-4 rounded-2xl cursor-pointer font-semibold transition-all border-2 ${
              isDragging
                ? 'border-indigo-500 bg-indigo-600/20'
                : 'border-white/10 bg-slate-800 hover:bg-slate-700'
            }`}
          >
            <Upload size={22} />
            {isDragging ? 'أفلت الملف هنا' : 'رفع ملف صوتي'}
            <input
              type="file"
              accept="audio/*"
              className="hidden"
              onChange={(e) => e.target.files && handleFileUpload(e.target.files[0])}
            />
          </label>
        </div>

        {/* Error */}
        {error && (
          <div className="mb-6 bg-red-900/30 border border-red-500/40 text-red-300 rounded-2xl px-6 py-4 text-center">
            {error}
          </div>
        )}

        {/* Waveform + Controls */}
        {(file || isRecording) && (
          <div className="bg-slate-900/80 border border-white/5 rounded-3xl p-8 mb-8 shadow-2xl">
            {file && (
              <p className="text-slate-500 text-sm mb-4 text-center truncate">
                📎 {file.name} &nbsp;·&nbsp; {(file.size / 1024).toFixed(1)} KB
              </p>
            )}
            <div ref={waveformRef} className="mb-6" />
            <div className="flex gap-4 justify-center flex-wrap">
              <button
                onClick={() => wavesurfer.current?.playPause()}
                disabled={!file || isRecording}
                className="flex items-center gap-2 px-6 py-3 bg-slate-800 hover:bg-slate-700 rounded-xl transition disabled:opacity-40"
              >
                {isPlaying ? <Pause size={20} /> : <Play size={20} />}
                {isPlaying ? 'إيقاف مؤقت' : 'تشغيل'}
              </button>

              <button
                onClick={predictEmotion}
                disabled={loading || !file || isRecording}
                className="flex items-center gap-2 px-10 py-3 bg-indigo-600 hover:bg-indigo-700 rounded-xl font-semibold transition shadow-lg shadow-indigo-900/40 disabled:opacity-60"
              >
                <Brain size={20} />
                {loading ? 'جاري التحليل...' : 'تحليل العاطفة'}
              </button>

              <button
                onClick={reset}
                className="flex items-center gap-2 px-6 py-3 bg-slate-800 hover:bg-slate-700 rounded-xl transition"
                title="بدء من جديد"
              >
                <RefreshCw size={20} />
              </button>
            </div>
          </div>
        )}

        {/* Results */}
        {prediction && (
          <div className="bg-slate-900/80 border border-white/5 rounded-3xl p-8 shadow-2xl">
            {/* Primary result */}
            <div className="text-center mb-10">
              <div className="text-8xl mb-4 drop-shadow-lg">{prediction.emoji}</div>
              <h2 className="text-4xl font-bold mb-1">
                {EMOTION_LABELS[prediction.emotion] ?? prediction.emotion}
              </h2>
              <p className="text-slate-400">الثقة: <span className="text-indigo-400 font-semibold">{prediction.confidence}%</span></p>
            </div>

            {/* Confidence bars */}
            <div className="space-y-3">
              <h3 className="text-slate-400 text-sm font-medium mb-4 text-center uppercase tracking-widest">توزيع الاحتمالات</h3>
              {sortedProbabilities.map(([code, pct]) => (
                <div key={code} className="flex items-center gap-4">
                  <span className="w-20 text-sm text-slate-400 text-right shrink-0">
                    {EMOTION_LABELS[code] ?? code}
                  </span>
                  <div className="flex-1 bg-slate-800 rounded-full h-2.5 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-700 ${EMOTION_COLORS[code] ?? 'bg-indigo-500'}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="w-12 text-sm text-slate-300 text-right shrink-0">{pct.toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Empty state */}
        {!file && !isRecording && !prediction && (
          <div className="text-center text-slate-600 mt-16 select-none">
            <AudioLines size={48} className="mx-auto mb-4 opacity-30" />
            <p className="text-lg">سجّل صوتك أو اسحب ملفاً صوتياً هنا للبدء</p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="text-center text-slate-700 text-xs py-8">
        EmoVoice · 2D-CNN on CREMA-D · 6 Emotions
      </footer>
    </div>
  );
}

export default App;
