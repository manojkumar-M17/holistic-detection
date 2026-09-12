"""Headless validation runner built on the production monitoring components."""

import json
import os
import time
from pathlib import Path
from typing import Any

import cv2

import config.config as cfg
from modules.behaviour import analyze_student_behaviour
from modules.camera import CameraManager
from modules.detection import StudentDetector, find_student_objects
from modules.holistic import HolisticDetector
from modules.suspicious_engine import SuspiciousEngine
from validation.evaluation import summarize_risk, summarize_stability


class ValidationInputError(ValueError):
    """Raised when a validation recording cannot be read."""


def _config_snapshot() -> dict[str, Any]:
    names = [
        "CAMERA_SOURCE",
        "FRAME_WIDTH",
        "FRAME_HEIGHT",
        "HEAD_YAW_THRESHOLD",
        "HEAD_PITCH_THRESHOLD",
        "SUSPICIOUS_DURATION",
        "HAND_TO_FACE_THRESHOLD",
        "SHOULDER_TILT_THRESHOLD",
        "ABSENCE_TIMEOUT_SECONDS",
        "AUDIO_NOISE_THRESHOLD",
        "RISK_DECAY_RATE",
        "SEVERITY_THRESHOLDS",
    ]
    return {name: getattr(cfg, name) for name in names}


def _events_from_reason(reason: str) -> list[str]:
    text = str(reason).upper()
    mappings = (
        ("CELL PHONE", "CELL_PHONE"),
        ("BOOK/NOTEBOOK", "BOOK_DETECTED"),
        ("BOOK", "BOOK_DETECTED"),
        ("LAPTOP", "LAPTOP_DETECTED"),
        ("MULTIPLE PERSON", "MULTIPLE_PERSON"),
        ("SPEECH", "TALKING"),
        ("WHISPER", "TALKING"),
        ("ABSENT", "STUDENT_ABSENT"),
        ("HAND NEAR FACE", "HAND_FACE"),
        ("STANDING", "STANDING"),
        ("LEANING", "LEANING"),
        ("LOOKING LEFT", "LOOK_LEFT"),
        ("LOOKING RIGHT", "LOOK_RIGHT"),
        ("LOOKING UP", "LOOK_UP"),
        ("LOOKING DOWN", "LOOK_DOWN"),
    )
    return list(dict.fromkeys(event for marker, event in mappings if marker in text))


class ValidationRunner:
    """Process a local recording through the same components as ``main.py``."""

    def __init__(self, video_path: str | os.PathLike[str], output_dir: str | os.PathLike[str] = "validation/results", model_path: str | None = None):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.model_path = model_path or cfg.YOLO_MODEL_PATH

    def run(self, session_name: str | None = None) -> dict[str, Any]:
        if not self.video_path.is_file():
            raise ValidationInputError(f"Video file does not exist: {self.video_path}")

        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            capture.release()
            raise ValidationInputError(f"Video file cannot be opened: {self.video_path}")

        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        camera = CameraManager(source=str(self.video_path), width=cfg.FRAME_WIDTH, height=cfg.FRAME_HEIGHT)
        detector = StudentDetector(model_path=self.model_path)
        holistic = HolisticDetector()
        engine = SuspiciousEngine()
        engine._trigger_alert = lambda *args, **kwargs: None
        session = session_name or self.video_path.stem
        detections: list[dict[str, Any]] = []
        frame_times: list[float] = []
        frames_processed = 0
        started = time.perf_counter()

        try:
            while True:
                ret, frame = camera.read_frame()
                if not ret or frame is None:
                    break
                frame_started = time.perf_counter()
                processed_bgr, _ = camera.preprocess_frame(frame, resize=True, blur=False)
                result = detector.detect_and_track(processed_bgr)
                students = result["students"]
                objects = result["objects"]
                timestamp = frames_processed / source_fps if source_fps > 0 else time.perf_counter() - started
                audio_metrics = None

                for student in students:
                    student_id = student["id"]
                    bbox = student["bbox"]
                    related_objects = find_student_objects(bbox, objects)
                    landmarks = holistic.process_student(processed_bgr, bbox, student_id)
                    features = analyze_student_behaviour(landmarks, related_objects)
                    suspicious, reason, risk, severity, category = engine.check_suspicious(
                        student_id,
                        features,
                        processed_bgr,
                        bbox,
                        total_student_count=len(students),
                        audio_metrics=audio_metrics,
                    )
                    confidence_values = [student.get("conf")]
                    confidence_values.extend(obj.get("conf") for obj in related_objects)
                    confidence_values = [value for value in confidence_values if value is not None]
                    if suspicious:
                        for event in _events_from_reason(reason):
                            detections.append({
                                "timestamp": round(timestamp, 4),
                                "student_id": student_id,
                                "event": event,
                                "risk_score": round(float(risk), 2),
                                "severity": severity,
                                "category": category,
                                "confidence": max(confidence_values) if confidence_values else None,
                                "reason": reason,
                            })

                frames_processed += 1
                frame_times.append(time.perf_counter() - frame_started)
        finally:
            camera.release()
            capture.release()
            holistic.release_all()

        if frames_processed == 0:
            raise ValidationInputError(
                f"Video contains no readable frames: {self.video_path}"
            )

        elapsed = time.perf_counter() - started
        duration = frames_processed / source_fps if source_fps > 0 else elapsed
        result = {
            "session": session,
            "video_file": str(self.video_path),
            "duration_seconds": round(duration, 4),
            "frames_processed": frames_processed,
            "source_fps": source_fps,
            "processing_seconds": round(elapsed, 4),
            "average_fps": round(frames_processed / elapsed, 3) if elapsed else 0.0,
            "minimum_frame_fps": round(1 / max(frame_times), 3) if frame_times else 0.0,
            "maximum_frame_fps": round(1 / min(frame_times), 3) if frame_times else 0.0,
            "configuration": _config_snapshot(),
            "detections": detections,
        }
        result["risk"] = summarize_risk(detections)
        result["stability"] = summarize_stability(detections)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{session}.json"
        result["result_file"] = str(output_path)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
