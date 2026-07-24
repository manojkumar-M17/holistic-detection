import time
import math
import numpy as np
import threading
import config.config as cfg

class AudioEngine:
    """
    Real-time audio monitoring engine that captures microphone audio
    and calculates RMS volume, spectral zero-crossing activity, and audio anomaly alerts.
    Falls back gracefully if pyaudio or sounddevice is not available.
    """
    def __init__(self, threshold=cfg.AUDIO_NOISE_THRESHOLD, sample_rate=cfg.AUDIO_SAMPLE_RATE):
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.is_running = False
        self.current_rms = 0.0
        self.volume_db = -60.0
        self.is_speech = False
        self.audio_alert = False
        self.lock = threading.Lock()
        self.thread = None
        self._backend = None

    def start(self):
        if not cfg.ENABLE_AUDIO_DETECTION or self.is_running:
            return
        
        self.is_running = True
        
        # Try importing sounddevice or pyaudio
        try:
            import sounddevice as sd
            self._backend = "sounddevice"
        except ImportError:
            try:
                import pyaudio
                self._backend = "pyaudio"
            except ImportError:
                self._backend = "fallback"

        self.thread = threading.Thread(target=self._listen_loop, name="Audio_Monitor_Thread", daemon=True)
        self.thread.start()
        print(f"[AUDIO] AudioEngine active using backend: {self._backend}")

    def _listen_loop(self):
        if self._backend == "sounddevice":
            import sounddevice as sd
            def audio_callback(indata, frames, time_info, status):
                if not self.is_running:
                    return
                audio_data = indata[:, 0]
                self._process_audio_chunk(audio_data)
            
            try:
                with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=audio_callback, blocksize=1024):
                    while self.is_running:
                        time.sleep(0.1)
            except Exception as e:
                print(f"[AUDIO] sounddevice stream error: {e}. Switching to fallback.")
                self._backend = "fallback"
                self._fallback_loop()

        elif self._backend == "pyaudio":
            import pyaudio
            try:
                p = pyaudio.PyAudio()
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=1024
                )
                while self.is_running:
                    data = stream.read(1024, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    self._process_audio_chunk(audio_data)
                    time.sleep(0.01)
                stream.stop_stream()
                stream.close()
                p.terminate()
            except Exception as e:
                print(f"[AUDIO] PyAudio error: {e}. Switching to fallback.")
                self._backend = "fallback"
                self._fallback_loop()
        else:
            self._fallback_loop()

    def _fallback_loop(self):
        """
        Fallback simulation loop when hardware mic libraries are not present.
        Maintains non-zero baseline RMS with low random ambient noise.
        """
        while self.is_running:
            # Simulated ambient noise baseline ~0.01 - 0.03
            simulated_chunk = np.random.normal(0, 0.015, 1024)
            self._process_audio_chunk(simulated_chunk)
            time.sleep(0.1)

    def _process_audio_chunk(self, audio_data):
        if len(audio_data) == 0:
            return
        
        rms = np.sqrt(np.mean(np.square(audio_data)))
        db = 20 * math.log10(max(rms, 1e-5))
        
        # Zero crossing rate (indicates speech / high pitch whispers)
        zcr = np.mean(np.abs(np.diff(np.signbit(audio_data))))
        
        speech_detected = (rms > self.threshold) and (zcr > 0.05)
        alert_triggered = rms > self.threshold

        with self.lock:
            self.current_rms = float(rms)
            self.volume_db = float(db)
            self.is_speech = speech_detected
            self.audio_alert = alert_triggered

    def get_metrics(self):
        with self.lock:
            return {
                "rms": round(self.current_rms, 4),
                "volume_db": round(self.volume_db, 1),
                "is_speech": self.is_speech,
                "audio_alert": self.audio_alert,
                "backend": self._backend
            }

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        print("[AUDIO] AudioEngine stopped.")
