import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_sqlite_db():
    print("\n--- 1. Testing SQLite Database & Review Pipeline ---")
    try:
        from database.db_manager import (
            init_db, log_incident, get_all_incidents,
            get_stats, update_incident_status, export_incidents_csv
        )
        init_db()
        print("[PASS] init_db() schema migration executed.")
        
        dummy_screenshot = "screenshots/dummy_test.jpg"
        os.makedirs("screenshots", exist_ok=True)
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(dummy_screenshot, dummy_img)
        
        incident_id = log_incident(
            student_id=99,
            activity="Unauthorized Item (Cell Phone)",
            screenshot_path=dummy_screenshot,
            severity="CRITICAL",
            category="PHONE_USAGE",
            risk_score=95.0
        )
        assert incident_id > 0, "Failed to log incident."
        print(f"[PASS] log_incident() created incident #{incident_id}.")
        
        # Test status update
        updated = update_incident_status(incident_id, "CONFIRMED")
        assert updated, "Failed to update incident status."
        print("[PASS] update_incident_status() set status to CONFIRMED.")
        
        incidents = get_all_incidents(student_id=99)
        assert len(incidents) > 0, "No incidents returned."
        assert incidents[0]["status"] == "CONFIRMED", "Incident status mismatch."
        print(f"[PASS] get_all_incidents() filtered and returned confirmed incident.")
        
        stats = get_stats()
        assert "severity_counts" in stats, "Stats missing severity counts."
        print(f"[PASS] get_stats() returned analytics: {stats['severity_counts']}")
        
        csv_report = export_incidents_csv()
        assert "Incident ID" in csv_report and "Cell Phone" in csv_report, "CSV report generation failed."
        print("[PASS] export_incidents_csv() successfully generated audit report.")
        
        if os.path.exists(dummy_screenshot):
            os.remove(dummy_screenshot)
            
        return True
    except Exception as e:
        print(f"[FAIL] SQLite DB test failed: {e}")
        return False

def test_camera():
    print("\n--- 2. Testing Camera Module ---")
    try:
        from modules.camera import CameraManager
        cam = CameraManager(source=0)
        ret, frame = cam.read_frame()
        if ret and frame is not None:
            print(f"[PASS] Camera captured live frame of size {frame.shape}")
        else:
            print("[INFO] Webcam source 0 not available. Creating mock frame.")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Mock Proctoring Frame", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        proc_bgr, proc_rgb = cam.preprocess_frame(frame)
        assert proc_bgr is not None and proc_rgb is not None, "Preprocessing failed"
        print(f"[PASS] Frame preprocessing completed. RGB shape: {proc_rgb.shape}")
        
        cam.release()
        return True
    except Exception as e:
        print(f"[FAIL] Camera test failed: {e}")
        return False

def test_mediapipe():
    print("\n--- 3. Testing MediaPipe Holistic ---")
    try:
        from modules.holistic import HolisticDetector
        hd = HolisticDetector()
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 540, 380)
        
        landmark_data = hd.process_student(mock_frame, bbox, 99)
        print(f"[PASS] MediaPipe Holistic processed student crop safely.")
        hd.release_all()
        return True
    except Exception as e:
        print(f"[FAIL] MediaPipe test failed: {e}")
        return False

def test_yolo_objects():
    print("\n--- 4. Testing YOLOv8 Forbidden Object & Tracking ---")
    try:
        from modules.detection import StudentDetector, find_student_objects
        sd = StudentDetector()
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        res = sd.detect_and_track(mock_frame)
        assert "students" in res and "objects" in res, "Detection result missing keys."
        print(f"[PASS] YOLOv8 ran multi-class detection safely. Students: {len(res['students'])}, Objects: {len(res['objects'])}")
        
        # Test spatial overlap helper
        student_bbox = (100, 100, 400, 400)
        detected_objects = [
            {"class_id": 67, "label": "Cell Phone", "bbox": (150, 150, 200, 250), "conf": 0.9},
            {"class_id": 73, "label": "Book/Notebook", "bbox": (600, 600, 700, 700), "conf": 0.8}
        ]
        correlated = find_student_objects(student_bbox, detected_objects)
        assert len(correlated) == 1 and correlated[0]["label"] == "Cell Phone", "Spatial correlation failed."
        print("[PASS] find_student_objects() correctly identified phone inside student ROI.")
        return True
    except Exception as e:
        print(f"[FAIL] YOLO object detection test failed: {e}")
        return False

def test_risk_engine():
    print("\n--- 5. Testing Risk Scoring & Behavioral Engine ---")
    try:
        from modules.behaviour import analyze_student_behaviour
        from modules.suspicious_engine import SuspiciousEngine
        
        engine = SuspiciousEngine()
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 400, 400)
        
        # Test 1: Correlated object -> Critical alert
        correlated_objs = [{"class_id": 67, "label": "Cell Phone", "bbox": (150, 150, 200, 200), "conf": 0.95}]
        features = analyze_student_behaviour(None, correlated_objects=correlated_objs)
        assert features["has_forbidden_object"], "Forbidden object flag not set."
        
        is_suspicious, reason, risk, severity, cat = engine.check_suspicious(1, features, mock_frame, bbox)
        assert is_suspicious and severity == "CRITICAL" and cat == "PHONE_USAGE", f"Unexpected risk output: {severity}, {cat}"
        print(f"[PASS] Risk Engine flagged CRITICAL alert for phone usage (Risk: {risk}%).")
        
        # Test 2: Risk score decay on normal behavior
        norm_features = analyze_student_behaviour(None)
        norm_features["is_absent"] = False
        _, _, decayed_risk, norm_sev, _ = engine.check_suspicious(1, norm_features, mock_frame, bbox)
        assert decayed_risk < risk, "Risk score decay failed."
        print(f"[PASS] Risk Engine correctly decayed score from {risk}% down to {decayed_risk}%.")
        
        return True
    except Exception as e:
        print(f"[FAIL] Risk Engine test failed: {e}")
        return False

def test_flask_endpoints():
    print("\n--- 6. Testing Dashboard REST API Endpoints ---")
    try:
        from dashboard.app import app
        client = app.test_client()
        
        res = client.get('/api/stats')
        assert res.status_code == 200 and "active_students" in res.get_json(), "API stats failed."
        print("[PASS] GET /api/stats endpoint returned 200 OK.")
        
        res = client.get('/api/alerts')
        assert res.status_code == 200, "API alerts failed."
        print("[PASS] GET /api/alerts endpoint returned 200 OK.")
        
        res = client.get('/api/config')
        assert res.status_code == 200 and "yaw_threshold" in res.get_json(), "API config failed."
        print("[PASS] GET /api/config endpoint returned 200 OK.")
        
        res = client.get('/api/incidents/export')
        assert res.status_code == 200 and "text/csv" in res.content_type, "API export failed."
        print("[PASS] GET /api/incidents/export returned CSV file.")
        
        return True
    except Exception as e:
        print(f"[FAIL] Dashboard API test failed: {e}")
        return False

def main():
    print("=== NEXT STAGE INTEGRITY VERIFICATION TEST ===")
    results = [
        ("SQLite DB & Review Test", test_sqlite_db()),
        ("Camera Module Test", test_camera()),
        ("MediaPipe Holistic Test", test_mediapipe()),
        ("YOLO Forbidden Object Test", test_yolo_objects()),
        ("Risk Scoring & Behavior Engine Test", test_risk_engine()),
        ("Dashboard REST API Test", test_flask_endpoints())
    ]
    
    print("\n=== VERIFICATION SUMMARY ===")
    all_passed = True
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
            
    if all_passed:
        print("\nAll Next Stage components verified successfully! System is 100% operational.")
        sys.exit(0)
    else:
        print("\nSome tests failed. Please review errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

