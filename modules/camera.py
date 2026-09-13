import cv2
import time

class CameraManager:
    def __init__(self, source=0, width=640, height=480):
        """
        Initializes the Video Capture stream.
        source: 0 for webcam, or video file path
        """
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self._last_open_attempt = 0.0
        self.open_stream()

    def open_stream(self):
        """
        Attempts to open the video capture stream.
        """
        self._last_open_attempt = time.time()
        if str(self.source).lower() in ("demo", "synthetic"):
            self.is_synthetic = True
            self._frame_count = 0
            print(f"[CAMERA] Synthetic/Demo source '{self.source}' initialized.")
            return

        self.is_synthetic = False
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f"[CAMERA] Error: Could not open source {self.source}")
        else:
            # Set properties (applies to cameras mostly)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            print(f"[CAMERA] Source {self.source} opened successfully.")

    def read_frame(self):
        """
        Reads a frame from the capture stream.
        """
        if getattr(self, "is_synthetic", False):
            self._frame_count += 1
            import numpy as np
            frame = np.full((self.height, self.width, 3), 25, dtype=np.uint8)
            # Draw synthetic desk background
            cv2.rectangle(frame, (80, 240), (self.width - 80, self.height - 20), (50, 50, 60), -1)
            cv2.rectangle(frame, (80, 240), (self.width - 80, self.height - 20), (80, 80, 95), 2)
            cv2.putText(frame, "EXAM DESK #1", (100, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 130), 1)
            # Simulation banner
            cv2.putText(
                frame, f"[DEMO SIMULATION] Frame {self._frame_count}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2
            )
            time.sleep(0.03)
            return True, frame

        if self.cap is None or not self.cap.isOpened():
            if time.time() - self._last_open_attempt > 2.0:
                self.open_stream()
            if self.cap is None or not self.cap.isOpened():
                return False, None
        
        ret, frame = self.cap.read()
        return ret, frame

    def preprocess_frame(self, frame, resize=True, blur=False):
        """
        Applies resizing, noise reduction, and returns the preprocessed BGR and RGB frames.
        """
        if frame is None:
            return None, None

        # 1. Resize if required
        if resize:
            processed_frame = cv2.resize(frame, (self.width, self.height))
        else:
            processed_frame = frame.copy()

        # 2. Noise reduction (Gaussian Blur) if required
        if blur:
            processed_frame = cv2.GaussianBlur(processed_frame, (3, 3), 0)

        # 3. RGB conversion (MediaPipe requires RGB)
        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        return processed_frame, rgb_frame

    def release(self):
        """
        Releases the Video Capture resources.
        """
        self.is_synthetic = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
            print("[CAMERA] Resources released.")
