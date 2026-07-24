import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Database and file logging configuration
DB_PATH = os.path.join(BASE_DIR, "database", "exam_monitoring.db")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Ensure directories exist
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# YOLO Model configuration
YOLO_MODEL_PATH = os.path.join(WEIGHTS_DIR, "yolov8n.pt")

# Camera configuration
# 0 for webcam, or absolute path to a video file
CAMERA_SOURCE = 0 
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Suspicious Activity Detection Thresholds
HEAD_YAW_THRESHOLD = 25  # Degrees for head turning left/right
HEAD_PITCH_THRESHOLD = 15  # Degrees for looking up/down
SUSPICIOUS_DURATION = 1.5  # Seconds of continuous behavior before alert

HAND_TO_FACE_THRESHOLD = 0.40  # Proximity ratio relative to face/shoulder dimension
SHOULDER_TILT_THRESHOLD = 20  # Angle in degrees for abnormal leaning posture
STANDING_Y_THRESHOLD_RATIO = 0.25  # Bounding box height shift/ratio check for standing

# Audio Anomaly & Whisper Detection Thresholds
ENABLE_AUDIO_DETECTION = True
AUDIO_NOISE_THRESHOLD = 0.08  # Normalized RMS audio threshold for whisper/speech alert
AUDIO_SAMPLE_RATE = 16000

# Proxy & Multi-Person Detection Thresholds
MAX_ALLOWED_STUDENTS = 1
ENABLE_PROXY_DETECTION = True
ABSENCE_TIMEOUT_SECONDS = 5.0

# Exam & Session Default Metadata
DEFAULT_EXAM_NAME = "Standard Proctored Exam"
DEFAULT_CANDIDATE_NAME = "Student_Candidate_01"

# COCO Object Classes for YOLOv8 Detection
PERSON_CLASS = 0
CELL_PHONE_CLASS = 67
BOOK_CLASS = 73
LAPTOP_CLASS = 63
FORBIDDEN_CLASSES = {
    CELL_PHONE_CLASS: "Cell Phone",
    BOOK_CLASS: "Book/Notebook",
    LAPTOP_CLASS: "Laptop"
}

# Feature Toggles & Risk Scoring Parameters
ENABLE_OBJECT_DETECTION = True
ENABLE_ABSENCE_DETECTION = True
RISK_DECAY_RATE = 2.0  # Risk score points reduced per frame of normal behavior
SEVERITY_THRESHOLDS = {
    "LOW": 25,
    "MEDIUM": 50,
    "HIGH": 75,
    "CRITICAL": 90
}

# Flask dashboard configuration
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000


