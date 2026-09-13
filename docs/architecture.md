# System Architecture & Technical Specification

The **AI Exam Hall Monitoring System** is an edge-first, multimodal automated proctoring platform designed to monitor exam halls and individual workstations in real time. It combines computer vision, deep-learning-based spatial object tracking, landmark-based gaze and posture analysis, acoustic anomaly surveillance, and stateful risk scoring to detect academic dishonesty while maintaining zero external cloud dependencies.

---

## 1. High-Level Architecture Overview

The system operates across three concurrent subsystems coordinated via thread-safe shared state and a local transactional database:

```mermaid
flowchart TD
    subgraph SENSORS ["Sensor Acquisition Layer"]
        CAM["Camera Stream<br/>(Webcam / Video File / Demo)"]
        MIC["Acoustic Stream<br/>(Microphone / PyAudio / Fallback)"]
    end

    subgraph VISION ["Computer Vision & Holistic Analysis"]
        YOLO["YOLOv8n Detector<br/>(Student & Forbidden Item Bounding Boxes)"]
        TRACK["ByteTrack / Centroid Tracker<br/>(Persistent Student IDs)"]
        HOLISTIC["MediaPipe Holistic<br/>(Face Mesh, Pose Skeleton, Hands)"]
        SPATIAL["Spatial Correlation Engine<br/>(Correlate Items to Student ROI)"]
        BEHAV["Behaviour Analysis<br/>(Gaze Vector, Posture Angle, Hand-to-Face)"]
    end

    subgraph AUDIO ["Acoustic Processing"]
        AUDIO_ENG["Audio Engine<br/>(RMS Calculation, Speech/Whisper Gate)"]
    end

    subgraph RISK ["Integrity & State Engine"]
        STUDENT_MGR["Student State Manager<br/>(Presence, Active Status, Lifecycle)"]
        ABSENCE["Absence Detector<br/>(Timeout & Disappearance Tracking)"]
        RISK_ENG["Suspicious Rule Engine<br/>(Weighted Scoring & Temporal Cooldown)"]
    end

    subgraph STORAGE ["Local Storage Layer (Zero Cloud)"]
        SQLITE[("SQLite Database<br/>(WAL Mode, Indexed Incidents)")]
        SCREENSHOTS["Encrypted Screenshot Vault<br/>(Incident Evidence Frames)"]
        REPORTS["Audit Report Generator<br/>(Sanitized HTML & Printable PDF)"]
    end

    subgraph DASHBOARD ["Presentation & Control Layer"]
        FLASK["Flask Web Server<br/>(Threaded, Werkzeug WSGI)"]
        MJPEG["MJPEG Video Stream<br/>(Overlaid Bounding Boxes & Skeletons)"]
        SSE["Server-Sent Events (SSE)<br/>(Real-Time Telemetry & Alert Feed)"]
        UI["Modern Web UI<br/>(Filters, Status Actions, Live Tuning)"]
    end

    CAM --> YOLO
    YOLO --> TRACK
    TRACK --> SPATIAL
    YOLO --> SPATIAL
    TRACK --> HOLISTIC
    HOLISTIC --> BEHAV
    SPATIAL --> BEHAV

    MIC --> AUDIO_ENG

    BEHAV --> RISK_ENG
    AUDIO_ENG --> RISK_ENG
    TRACK --> STUDENT_MGR
    STUDENT_MGR --> ABSENCE
    ABSENCE --> RISK_ENG

    RISK_ENG --> SQLITE
    RISK_ENG --> SCREENSHOTS
    SQLITE --> REPORTS

    CAM --> MJPEG
    RISK_ENG --> SSE
    STUDENT_MGR --> SSE
    AUDIO_ENG --> SSE
    FLASK --> UI
    MJPEG --> UI
    SSE --> UI
    REPORTS --> UI
```

---

## 2. Core Processing Pipeline

The monitoring loop operates as a background worker thread (`AI_Core_Thread`) executing the following pipeline for every video frame:

```mermaid
sequenceDiagram
    autonumber
    participant Camera as CameraManager
    participant Detector as StudentDetector (YOLO)
    participant Holistic as HolisticDetector (MediaPipe)
    participant Behaviour as BehaviourAnalyzer
    participant Risk as SuspiciousEngine
    participant Database as SQLite (WAL)
    participant State as SharedState / Flask

    Camera->>Camera: Capture & Preprocess Frame (640x480)
    Camera->>Detector: Run Detection & Object Association
    Detector-->>Camera: Student BBoxes & Restricted Items (Phone, Book, Laptop)
    
    loop For each detected student
        Camera->>Holistic: Crop ROI & Run Holistic Mesh
        Holistic-->>Behaviour: 468 Face, 33 Pose, 42 Hand Landmarks
        Behaviour-->>Risk: Extracted Features (Yaw, Pitch, Hand-Face, Tilt)
        Risk->>Risk: Temporal Filter & Dynamic Scoring
        alt Suspicious Incident Triggered & Cooldown Expired
            Risk->>Camera: Capture Evidence Screenshot
            Risk->>Database: Persist Incident Record (Severity, Activity, Risk %)
        end
    end

    opt Established Student Disappeared > 5.0s
        Risk->>Risk: Evaluate Absence Trigger (bbox=None)
        Risk->>Database: Persist STUDENT_ABSENT Incident
    end

    Risk->>State: Publish Frame, Metrics & Alert Telemetry
    State->>State: Stream to Web Dashboard via SSE & MJPEG
```

---

## 3. Subsystem Breakdown

### 3.1 Camera & Frame Ingestion (`modules/camera.py`)
- **Webcam / USB Feed**: Interfaced via OpenCV `cv2.VideoCapture`.
- **Pre-recorded Video Files**: Supports `.mp4`, `.avi`, `.mov` for deterministic evaluation.
- **Deterministic Synthetic Demo Mode (`source='demo'`)**: Generates calibrated synthetic classroom frames at ~25 FPS with candidate representations, desk geometry, and simulation banners.
- **Resilience**: Auto-reconnects if camera hardware disconnects; falls back gracefully if frame capture fails.

### 3.2 Student & Restricted Object Detection (`modules/detection.py`)
- **Model**: YOLOv8n (Ultralytics nano architecture).
- **Target Classes**:
  - `0`: Person (Students in exam hall)
  - `67`: Cell Phone (High-risk cheating device)
  - `73`: Book / Notebook (Unauthorized reference material)
  - `63`: Laptop / Screen (Secondary electronic screen)
- **Spatial Correlation**: Bounding boxes of forbidden objects are spatially intersected with student bounding boxes using IoU and relative centroid proximity (`find_student_objects`).

### 3.3 Facial Landmark & Pose Analysis (`modules/holistic.py`, `modules/behaviour.py`)
- **MediaPipe Holistic**:
  - **468 Face Mesh Landmarks**: Computes 3D head yaw (left/right turn) and pitch (looking down at notes or up at ceiling) using eye corners, nose tip, and chin landmarks.
  - **33 Body Pose Landmarks**: Tracks shoulder slope (`shoulder_tilt`) for abnormal leaning, and hip/shoulder ratio for standing up.
  - **42 Hand Landmarks**: Measures Euclidean distance between wrist/finger tips and facial landmarks to detect hand-to-face concealment or whispering.

### 3.4 Acoustic Surveillance (`modules/audio_engine.py`)
- **Capture Backends**: Hardware capture via `sounddevice` or `pyaudio`, with automated fallback to simulated baseline noise if audio libraries are unavailable.
- **RMS Volume & Speech Gating**: Computes Root Mean Square (RMS) energy and zero-crossing frequency in decibels (dB).
- **Adaptive Calibration**: Measures ambient noise baseline during initial 2.0 seconds to automatically adjust trigger threshold relative to room acoustics.

### 3.5 Absence Detection Engine (`modules/suspicious_engine.py`, `main.py`)
- **Lifecycle Tracking**: `established_students` tracks students confirmed across at least 5 frames.
- **Absence Timeout**: When an established student is absent from subsequent frames for $> \text{ABSENCE\_TIMEOUT\_SECONDS}$ (5.0s default), `is_absent=True` is triggered.
- **Safe Coordinate Handling**: If `bbox is None`, screenshot alert rendering applies an edge-to-edge warning banner across the frame header without coordinate unpacking errors.
- **State Recovery**: When the student re-enters the frame, the absence duration timer resets and risk score decays gracefully at `RISK_DECAY_RATE` (2.0% per frame) back to zero.

### 3.6 Risk Assessment & Rule Engine (`modules/suspicious_engine.py`, `modules/rule_engine.py`)
- **Multi-Level Severity**:
  - `CRITICAL` ($\ge 90\%$): Cell phone detection, multiple persons / proxy.
  - `HIGH` ($\ge 75\%$): Persistent head turns, laptops, prolonged absence.
  - `MEDIUM` ($\ge 50\%$): Whispering, books, abnormal leaning.
  - `LOW` ($\ge 25\%$): Fleeting gaze shifts, brief hand-near-face.
  - `NORMAL` ($< 25\%$): Standard examination behavior.
- **Temporal Event Cooldowns**: Independent cooldowns per incident type (`EVENT_COOLDOWNS`) prevent flooding the database while an anomalous behavior persists.

### 3.7 Local Persistence & Reporting (`database/db_manager.py`, `modules/report_generator.py`)
- **SQLite Database with WAL**: Configured with `PRAGMA journal_mode = WAL;` and a 10.0s connection timeout for safe concurrent reads from Flask and writes from the detection thread.
- **Indexes**: Indexes on `student_id`, `timestamp`, `severity`, and `status` ensure $O(\log n)$ filter lookups.
- **HTML/PDF Audit Reports**: Generates formal proctoring audit summaries with integrity scores and incident breakdown. All user parameters (`exam_name`, `candidate_name`) and DB fields are sanitized via `html.escape()` to prevent XSS injection.

---

## 4. Threading & Concurrency Model

```mermaid
flowchart LR
    subgraph MAIN ["Process: Python main.py"]
        T1["Thread 1: AI_Core_Thread<br/>- Frame acquisition<br/>- YOLOv8 inference<br/>- MediaPipe Holistic<br/>- Risk assessment<br/>- DB Incident logging"]
        T2["Thread 2: Flask_Server_Thread<br/>- HTTP API endpoints<br/>- MJPEG video generator<br/>- SSE event stream<br/>- CSV / HTML report export"]
        T3["Thread 3: Audio_Monitor_Thread<br/>- Continuous mic stream<br/>- RMS & spectral analysis<br/>- Baseline adaptation"]
    end

    subgraph SYNC ["Thread Synchronization"]
        EV["shutdown_event (threading.Event)"]
        STATE["SharedState (Atomic assignments)"]
        LOCK["SQLite WAL + Connection Timeouts"]
    end

    T1 --> STATE
    T3 --> STATE
    STATE --> T2
    EV --> T1
    EV --> T2
    EV --> T3
    T1 --> LOCK
    T2 --> LOCK
```

---

## 5. Security & Privacy Guarantees

1. **Local-Only Processing**: All video, audio, and database operations run entirely on the local machine (`127.0.0.1`). Zero data is transmitted to external servers or cloud APIs.
2. **Deterministic Data Deletion**: All logs and database tables can be wiped instantly via the dashboard API (`POST /api/incidents/clear`) or manual deletion of `database/exam_monitoring.db`.
3. **No Unencrypted Remote Transmission**: In production deployment behind an institutional gateway, Flask should be placed behind a reverse proxy (e.g. NGINX with TLS/HTTPS).
4. **Input Sanitization**: All reporting components apply HTML entity escaping to eliminate stored and reflected XSS vulnerabilities.

