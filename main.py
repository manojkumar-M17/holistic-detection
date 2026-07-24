import cv2
import threading
import time
import sys
import os
import numpy as np

# Add parent directory to path to ensure relative imports work when executing main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import CAMERA_SOURCE, FRAME_WIDTH, FRAME_HEIGHT
from database.db_manager import init_db
from modules.camera import CameraManager
from modules.detection import StudentDetector, find_student_objects
from modules.holistic import HolisticDetector
from modules.behaviour import analyze_student_behaviour
from modules.suspicious_engine import SuspiciousEngine
from modules.audio_engine import AudioEngine
from dashboard.app import SharedState, run_flask_server

def draw_student_overlays(frame, bbox, student_id, landmark_data, features, is_suspicious, suspicion_reason, risk_score=0.0, severity="NORMAL"):
    """
    Draws custom stylized overlays (bounding box, skeletons, hands, gaze pointer, risk score badges)
    directly onto the main camera frame.
    """
    x1, y1, x2, y2 = bbox
    
    # 1. Color scheme based on severity level
    color_map = {
        "CRITICAL": (0, 0, 255),
        "HIGH": (0, 69, 255),
        "MEDIUM": (0, 165, 255),
        "LOW": (0, 255, 255),
        "NORMAL": (0, 255, 0)
    }
    box_color = color_map.get(severity, (0, 255, 0))
    text_color = (255, 255, 255)
    
    # Draw student bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    
    # Header bar
    header_h = 24
    cv2.rectangle(frame, (x1, max(0, y1 - header_h)), (x2, y1), box_color, -1)
    
    # Header badge text
    short_reason = suspicion_reason.split(',')[0]
    status_label = f"ID: {student_id} | Risk: {risk_score:.0f}% | {short_reason}"
    cv2.putText(
        frame, status_label, (x1 + 6, max(14, y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA
    )
    
    if landmark_data is None or features is None:
        return
        
    offset_x, offset_y, crop_w, crop_h = landmark_data["crop_dims"]
    
    def local_to_global(lm):
        gx = int(offset_x + lm.x * crop_w)
        gy = int(offset_y + lm.y * crop_h)
        return gx, gy

    # 2. Draw Pose Skeleton (Shoulders and Hips)
    pose = landmark_data["pose_landmarks"]
    if pose is not None:
        try:
            pt11 = local_to_global(pose.landmark[11])
            pt12 = local_to_global(pose.landmark[12])
            
            cv2.line(frame, pt11, pt12, (255, 255, 0), 2)
            cv2.circle(frame, pt11, 4, (255, 0, 255), -1)
            cv2.circle(frame, pt12, 4, (255, 0, 255), -1)
            
            if pose.landmark[23].visibility > 0.5 and pose.landmark[24].visibility > 0.5:
                pt23 = local_to_global(pose.landmark[23])
                pt24 = local_to_global(pose.landmark[24])
                
                cv2.line(frame, pt23, pt24, (255, 255, 0), 2)
                cv2.line(frame, pt11, pt23, (255, 255, 0), 2)
                cv2.line(frame, pt12, pt24, (255, 255, 0), 2)
                cv2.circle(frame, pt23, 4, (255, 0, 255), -1)
                cv2.circle(frame, pt24, 4, (255, 0, 255), -1)
        except Exception:
            pass

    # 3. Draw Hands Landmarks (21 points per hand)
    for hand_lms in [landmark_data["left_hand_landmarks"], landmark_data["right_hand_landmarks"]]:
        if hand_lms is not None:
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16),
                (13, 17), (17, 18), (18, 19), (19, 20),
                (0, 17)
            ]
            for conn in connections:
                try:
                    pt1 = local_to_global(hand_lms.landmark[conn[0]])
                    pt2 = local_to_global(hand_lms.landmark[conn[1]])
                    cv2.line(frame, pt1, pt2, (0, 255, 255), 1)
                except Exception:
                    pass
            for lm in hand_lms.landmark:
                try:
                    pt = local_to_global(lm)
                    cv2.circle(frame, pt, 3, (255, 165, 0), -1)
                except Exception:
                    pass

    # 4. Draw Nose Gaze Vector Line
    gaze_line = features["gaze_line"]
    if gaze_line is not None:
        try:
            start_lm, end_pt = gaze_line
            start_pt = local_to_global(start_lm)
            
            end_gx = int(offset_x + end_pt[0])
            end_gy = int(offset_y + end_pt[1])
            
            cv2.arrowedLine(frame, start_pt, (end_gx, end_gy), (255, 0, 0), 2, tipLength=0.25)
        except Exception:
            pass

def draw_object_overlays(frame, detected_objects):
    """
    Draws highlighted bounding boxes and labels for forbidden items (Cell phones, Books, Laptops).
    """
    for obj in detected_objects:
        ox1, oy1, ox2, oy2 = obj["bbox"]
        label = f"FORBIDDEN: {obj['label']} ({obj['conf']*100:.0f}%)"
        
        cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 0, 255), 3)
        cv2.rectangle(frame, (ox1, max(0, oy1 - 20)), (ox2, oy1), (0, 0, 255), -1)
        cv2.putText(
            frame, label, (ox1 + 4, max(14, oy1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )

def processing_loop():
    """
    Background worker thread running YOLOv8 tracking, object correlation,
    MediaPipe Holistic, audio anomaly detection, risk scoring, DB alert logging, and stream updates.
    """
    print("[CORE] Initializing camera manager...")
    camera = CameraManager(source=CAMERA_SOURCE, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    
    print("[CORE] Initializing student & object detector engine...")
    detector = StudentDetector()
    
    print("[CORE] Initializing MediaPipe holistic engine...")
    holistic = HolisticDetector()
    
    print("[CORE] Initializing risk scoring logic engine...")
    engine = SuspiciousEngine()
    SharedState.suspicious_engine_ref = engine
    
    print("[CORE] Initializing audio anomaly detection engine...")
    audio_engine = AudioEngine()
    audio_engine.start()
    
    frame_count = 0
    cached_student_features = {}

    print("[CORE] Proctoring monitoring loop active.")
    
    try:
        while True:
            if not SharedState.monitoring_active:
                time.sleep(0.1)
                continue
                
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                time.sleep(0.03)
                continue
            
            frame_count += 1
            processed_bgr, processed_rgb = camera.preprocess_frame(frame, resize=True, blur=False)
            
            # 1. Multi-Student & Forbidden Object Detection
            detection_res = detector.detect_and_track(processed_bgr)
            students = detection_res["students"]
            all_objects = detection_res["objects"]
            
            total_students = len(students)
            SharedState.active_student_count = total_students
            current_alerts_count = 0
            annotated_frame = processed_bgr.copy()
            
            # Get latest microphone / audio telemetry
            audio_metrics = audio_engine.get_metrics()
            SharedState.audio_metrics = audio_metrics
            
            # 2. Draw Standalone Forbidden Object Highlights
            draw_object_overlays(annotated_frame, all_objects)
            
            current_ids = set()
            
            for student in students:
                sid = student["id"]
                bbox = student["bbox"]
                current_ids.add(sid)
                
                # Spatial correlation: find forbidden objects near/inside this student ROI
                correlated_objs = find_student_objects(bbox, all_objects)
                
                should_run_holistic = (frame_count % 2 == 0) or (sid not in cached_student_features)
                landmark_data = None
                features = None
                
                if should_run_holistic:
                    landmark_data = holistic.process_student(processed_bgr, bbox, sid)
                    features = analyze_student_behaviour(landmark_data, correlated_objects=correlated_objs)
                    cached_student_features[sid] = {
                        "landmarks": landmark_data,
                        "features": features
                    }
                else:
                    cache = cached_student_features.get(sid)
                    if cache:
                        landmark_data = cache["landmarks"]
                        features = cache["features"]
                        # Update correlated objects in cached features
                        features["has_forbidden_object"] = len(correlated_objs) > 0
                        features["forbidden_objects"] = [o["label"] for o in correlated_objs]

                # 3. Risk Score & Suspicious Behavior Check
                is_suspicious = False
                suspicion_reason = "Normal"
                risk_score = 0.0
                severity = "NORMAL"
                category = "GENERAL"
                
                is_suspicious, suspicion_reason, risk_score, severity, category = engine.check_suspicious(
                    sid, features, processed_bgr, bbox,
                    total_student_count=total_students,
                    audio_metrics=audio_metrics
                )
                if is_suspicious:
                    current_alerts_count += 1
                        
                # 4. Render Overlays with Risk Badges
                draw_student_overlays(
                    annotated_frame, bbox, sid, landmark_data, features,
                    is_suspicious, suspicion_reason, risk_score=risk_score, severity=severity
                )
            
            # Clean up stale cache
            stale_cache_ids = [sid for sid in cached_student_features if sid not in current_ids]
            for sid in stale_cache_ids:
                if sid in cached_student_features:
                    del cached_student_features[sid]
                    
            SharedState.active_alert_count = current_alerts_count
            SharedState.current_frame = annotated_frame
            SharedState.student_risk_scores = engine.risk_scores
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("[CORE] Monitoring loop stopped.")
    except Exception as e:
        print(f"[CORE] Exception in loop: {e}")
    finally:
        camera.release()
        holistic.release_all()
        audio_engine.stop()


def main():
    print("[MAIN] Initializing SQLite database...")
    init_db()
    
    proc_thread = threading.Thread(target=processing_loop, name="AI_Core_Thread")
    proc_thread.daemon = True
    proc_thread.start()
    
    print("[MAIN] Launching Flask Dashboard Server on http://127.0.0.1:5000")
    run_flask_server()

if __name__ == "__main__":
    main()

