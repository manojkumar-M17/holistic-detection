# AI Exam Hall Monitoring System

Holistic Detection is a Python application for monitoring an exam hall from a camera feed. It detects students and restricted objects, analyzes posture and gaze, records suspicious incidents in SQLite, and exposes a Flask dashboard for live review.

## Features

- YOLOv8 student/person and restricted-object detection
- Student tracking with stable IDs and bounding boxes
- MediaPipe Holistic face, pose, hand, and gaze analysis
- Suspicious behavior detection for gaze direction, phones, books, laptops, multiple people, standing, leaning, hands near face, talking, and absence
- Risk scores, severity levels, cooldowns, and incident logging
- Evidence screenshots and video recording support
- Flask dashboard with MJPEG video, Server-Sent Events, alert filters, status updates, CSV export, and HTML report generation
- Runtime configuration for detection thresholds and feature toggles

## Architecture

`main.py` initializes the database, starts the background monitoring thread, and runs the Flask server. The monitoring thread reads frames through `CameraManager`, runs `StudentDetector`, processes each student with `HolisticDetector`, evaluates behavior with `SuspiciousEngine`, stores incidents through the database layer, and publishes annotated frames through `dashboard/app.py`.

## Installation

Python 3.12.3 is the tested interpreter.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The tested computer-vision versions are MediaPipe `0.10.21`, OpenCV `4.11.0`, and NumPy `1.26.4`. Keep NumPy below 2 for MediaPipe compatibility.

## YOLO Model Setup

Place the YOLOv8 model at `weights/yolov8n.pt`. The path is configured by `YOLO_MODEL_PATH` in `config/config.py`. If the model cannot be loaded, the detector logs the problem and returns empty detections so the dashboard can still start.

## Camera Setup

Set `CAMERA_SOURCE`, `FRAME_WIDTH`, and `FRAME_HEIGHT` in `config/config.py`. `CAMERA_SOURCE` may be `0` for the default webcam or a video-file path. An unavailable camera is reported by `CameraManager`; tests use mocked capture objects and do not require a physical camera or GUI.

## Run the Application

```bash
source .venv/bin/activate
python main.py
```

Open <http://127.0.0.1:5000> in a browser. Stop the process with `Ctrl+C`.

## Run Tests

```bash
python -m pytest -q
python -m pip check
```

The test suite mocks camera, GUI, YOLO, and MediaPipe boundaries where appropriate. The live demo scripts under `tests/` only run when executed directly.

## Dashboard Usage

The dashboard provides the live feed, active student and alert metrics, alert filtering, incident status updates, CSV export, printable HTML report generation, screenshot serving, monitoring pause/resume controls, and runtime threshold configuration.

Important endpoints include:

- `/` dashboard page
- `/video_feed` MJPEG stream
- `/api/stream-events` live event stream
- `/api/stats` current metrics
- `/api/alerts` filtered incidents
- `/api/incidents/export` CSV export
- `/api/report/pdf` printable report response
- `/api/config` runtime configuration

## Database and Evidence

The SQLite database is `database/exam_monitoring.db`. The `incidents` table stores student ID, activity, timestamp, screenshot path, severity, category, risk score, and review status. Database directories and schema migrations are created automatically.

Screenshots are written to `screenshots/`, reports to `reports/`, and logs to `logs/`. These runtime directories are excluded from Git.

## Configuration

Detection thresholds, camera settings, feature toggles, risk thresholds, cooldowns, model paths, and Flask host/port are defined in `config/config.py`. The dashboard can update selected thresholds while the application is running.

## Troubleshooting

- **MediaPipe import errors:** install the pinned `mediapipe==0.10.21` and keep NumPy at `1.26.4`.
- **No camera:** verify `CAMERA_SOURCE`, permissions, and whether another application owns the device. The application will report an unavailable source.
- **No detections:** verify that `weights/yolov8n.pt` exists and is readable.
- **Port already in use:** change `FLASK_PORT` in `config/config.py`.
- **Database path tests:** the database manager recreates its cached manager when the configured `DB_PATH` changes.

## Known Limitations

- Detection quality depends on camera placement, lighting, model confidence, and available compute.
- Audio detection depends on the host microphone and supported audio backends.
- The default application is configured for a single local camera and a local Flask server.
- The generated report endpoint returns printable HTML; PDF conversion is not performed by the application itself.

## Real-World Validation

Phase 4 validation tools measure the existing production pipeline on local recordings without uploading video. See [validation/README.md](validation/README.md) for the controlled 5–15 minute recording workflow, annotation format, temporal tolerance, metrics, stability analysis, threshold experiments, and privacy guidance.

Run one recording with:

```bash
python tools/run_validation.py --video validation/recordings/session_01.mp4 --annotations validation/annotations/session_01.json
```

Run a small yaw-threshold comparison with:

```bash
python tools/tune_validation.py --video validation/recordings/session_01.mp4 --annotations validation/annotations/session_01.json --yaw-thresholds 20 25 30
```

The runner writes structured JSON and an HTML report under `validation/results/`. Tuning also writes `summary.json` and `summary.html`. The dashboard exposes `/api/validation/results` and `/api/validation/latest` for local summaries.

Synthetic tests verify the evaluation workflow only. Real-world precision, recall, and F1 remain pending an annotated recording. No real-world accuracy claim should be made without matching ground truth.