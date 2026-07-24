import os
import time
import json
import cv2
from flask import Flask, render_template, Response, jsonify, request, make_response, send_from_directory
import config.config as cfg
from database.db_manager import (
    get_all_incidents, get_stats, update_incident_status,
    export_incidents_csv, clear_all_incidents
)
from modules.report_generator import generate_proctoring_report

app = Flask(__name__)

# Global state sharing class
class SharedState:
    current_frame = None          # Stores the annotated frame for MJPEG stream
    active_student_count = 0      # Number of currently tracked students
    active_alert_count = 0        # Number of active alerts in current frame
    student_risk_scores = {}      # Student ID -> risk score percentage
    monitoring_active = True      # Toggle flag
    audio_metrics = {}            # Stores RMS, volume_db, speech flag
    suspicious_engine_ref = None  # Reference to active SuspiciousEngine instance

def gen_frames():
    """
    Generator function that yields JPEG frames for the video feed.
    """
    while True:
        if SharedState.monitoring_active and SharedState.current_frame is not None:
            ret, buffer = cv2.imencode('.jpg', SharedState.current_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            import numpy as np
            black_img = np.zeros((480, 640, 3), dtype=np.uint8)
            msg = "MONITORING PAUSED" if not SharedState.monitoring_active else "INITIALIZING Surveillance..."
            cv2.putText(
                black_img, msg, (120, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            ret, buffer = cv2.imencode('.jpg', black_img)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.04) # ~25 fps limit

def event_stream():
    """
    SSE stream pushing real-time metrics and alerts to connected proctoring web pages.
    """
    last_sent_alert_id = 0
    while True:
        db_stats = get_stats()
        all_inc = get_all_incidents()
        newest_alert = all_inc[0] if len(all_inc) > 0 else None
        
        event_data = {
            "timestamp": time.time(),
            "active_students": SharedState.active_student_count,
            "active_alerts": SharedState.active_alert_count,
            "total_logged_alerts": db_stats["total_alerts"],
            "student_risk_scores": SharedState.student_risk_scores,
            "audio_metrics": SharedState.audio_metrics,
            "new_alert": newest_alert if (newest_alert and newest_alert["id"] > last_sent_alert_id) else None
        }
        
        if newest_alert and newest_alert["id"] > last_sent_alert_id:
            last_sent_alert_id = newest_alert["id"]
            
        yield f"data: {json.dumps(event_data)}\n\n"
        time.sleep(1.0)

@app.route('/')
def index():
    """
    Renders the main proctoring monitoring dashboard.
    """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    Streams the live video feed using multipart MJPEG.
    """
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stream-events')
def stream_events():
    """
    Server-Sent Events (SSE) route for real-time live alert and telemetry updates.
    """
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/api/stats')
def api_stats():
    """
    Returns real-time system statistics, risk scores, and audio metrics.
    """
    db_stats = get_stats()
    return jsonify({
        "active_students": SharedState.active_student_count,
        "active_alerts": SharedState.active_alert_count,
        "total_logged_alerts": db_stats["total_alerts"],
        "unique_students_flagged": db_stats["unique_students"],
        "severity_counts": db_stats.get("severity_counts", {}),
        "status_counts": db_stats.get("status_counts", {}),
        "category_counts": db_stats.get("category_counts", {}),
        "student_risk_scores": SharedState.student_risk_scores,
        "audio_metrics": SharedState.audio_metrics
    })

@app.route('/api/alerts')
def api_alerts():
    """
    Returns filtered list of logged incidents from SQLite.
    """
    student_id = request.args.get('student_id')
    category = request.args.get('category', 'ALL')
    severity = request.args.get('severity', 'ALL')
    status = request.args.get('status', 'ALL')
    
    incidents = get_all_incidents(
        student_id=student_id,
        category=category,
        severity=severity,
        status=status
    )
    return jsonify(incidents)

@app.route('/api/incidents/<int:incident_id>/status', methods=['POST'])
def api_update_status(incident_id):
    """
    Updates the status of an incident ('UNREVIEWED', 'CONFIRMED', 'DISMISSED').
    """
    data = request.get_json() or {}
    new_status = data.get("status")
    if not new_status:
        return jsonify({"error": "Missing status parameter"}), 400
        
    success = update_incident_status(incident_id, new_status)
    if success:
        return jsonify({"status": "success", "incident_id": incident_id, "new_status": new_status})
    return jsonify({"error": "Incident not found or invalid status"}), 404

@app.route('/api/incidents/export')
def api_export_csv():
    """
    Exports incident log database as a downloadable CSV file.
    """
    csv_data = export_incidents_csv()
    response = make_response(csv_data)
    response.headers["Content-Disposition"] = "attachment; filename=proctoring_incident_report.csv"
    response.headers["Content-type"] = "text/csv"
    return response

@app.route('/api/report/pdf')
def api_download_report():
    """
    Generates and returns printable HTML/PDF proctoring summary report.
    """
    exam_name = request.args.get('exam_name', cfg.DEFAULT_EXAM_NAME)
    candidate_name = request.args.get('candidate_name', cfg.DEFAULT_CANDIDATE_NAME)
    filepath, html_content = generate_proctoring_report(exam_name, candidate_name)
    response = make_response(html_content)
    response.headers["Content-Disposition"] = "inline; filename=Proctoring_Audit_Report.html"
    response.headers["Content-type"] = "text/html"
    return response

@app.route('/api/incidents/clear', methods=['POST'])
def api_clear_incidents():
    """
    Clears all recorded incident logs from DB.
    """
    clear_all_incidents()
    return jsonify({"status": "success", "message": "All incident logs cleared"})

@app.route('/api/config', methods=['GET', 'POST'])
def api_manage_config():
    """
    Gets or updates detection thresholds live.
    """
    if request.method == 'POST':
        data = request.get_json() or {}
        if "yaw" in data:
            cfg.HEAD_YAW_THRESHOLD = float(data["yaw"])
        if "pitch" in data:
            cfg.HEAD_PITCH_THRESHOLD = float(data["pitch"])
        if "duration" in data:
            cfg.SUSPICIOUS_DURATION = float(data["duration"])
        if "enable_object_detection" in data:
            cfg.ENABLE_OBJECT_DETECTION = bool(data["enable_object_detection"])
        if "enable_absence_detection" in data:
            cfg.ENABLE_ABSENCE_DETECTION = bool(data["enable_absence_detection"])
        if "enable_audio_detection" in data:
            cfg.ENABLE_AUDIO_DETECTION = bool(data["enable_audio_detection"])
        if "audio_threshold" in data:
            cfg.AUDIO_NOISE_THRESHOLD = float(data["audio_threshold"])
            
        if SharedState.suspicious_engine_ref:
            SharedState.suspicious_engine_ref.update_thresholds(
                head_yaw_threshold=cfg.HEAD_YAW_THRESHOLD,
                head_pitch_threshold=cfg.HEAD_PITCH_THRESHOLD,
                audio_noise_threshold=cfg.AUDIO_NOISE_THRESHOLD
            )
            
    return jsonify({
        "yaw_threshold": cfg.HEAD_YAW_THRESHOLD,
        "pitch_threshold": cfg.HEAD_PITCH_THRESHOLD,
        "suspicious_duration": cfg.SUSPICIOUS_DURATION,
        "enable_object_detection": cfg.ENABLE_OBJECT_DETECTION,
        "enable_absence_detection": cfg.ENABLE_ABSENCE_DETECTION,
        "enable_audio_detection": cfg.ENABLE_AUDIO_DETECTION,
        "audio_threshold": cfg.AUDIO_NOISE_THRESHOLD
    })

@app.route('/screenshots/<path:filename>')
def get_screenshot(filename):
    """
    Serves saved screenshots.
    """
    return send_from_directory(cfg.SCREENSHOT_DIR, filename)

@app.route('/api/control/<action>')
def control_monitoring(action):
    """
    API endpoint to pause or resume monitoring.
    """
    if action == "pause":
        SharedState.monitoring_active = False
        return jsonify({"status": "paused"})
    elif action == "resume":
        SharedState.monitoring_active = True
        return jsonify({"status": "running"})
    return jsonify({"status": "unknown"})

def run_flask_server():
    """
    Starts the Flask app on the configured port.
    """
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    app.run(host=cfg.FLASK_HOST, port=cfg.FLASK_PORT, debug=False, threaded=True)


