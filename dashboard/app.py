import time
import json
import os
import logging
import math
import sqlite3
import hmac
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, render_template, Response, jsonify, request, make_response, send_from_directory
from werkzeug.exceptions import HTTPException
from datetime import date, datetime
from enum import Enum
import config.config as cfg
from database.db_manager import (
    get_all_incidents, get_manager, get_stats, init_db, update_incident_status,
    export_incidents_csv, clear_all_incidents
)
from modules.report_generator import generate_proctoring_report

app = Flask(__name__)
logger = logging.getLogger(__name__)

FRONTEND_ORIGIN = os.environ.get(
    "FRONTEND_ORIGIN",
    "https://holistic-detection.vercel.app",
).strip().rstrip("/")

# Gunicorn imports this module directly; main.py's initialization is not run in that deployment.
DATABASE_SCHEMA_INITIALIZED = bool(init_db())
logger.info("Initialized Flask database schema")

WRITE_ENDPOINTS = {
    "api_update_status",
    "api_clear_incidents",
    "api_manage_config",
    "control_monitoring",
}

READ_ENDPOINTS = {
    "stream_events",
    "api_stats",
    "api_alerts",
    "api_export_csv",
    "api_download_report",
    "api_manage_config",
    "api_validation_results",
    "api_validation_latest",
    "get_screenshot",
    "video_feed",
}

PROTECTED_ENDPOINTS = WRITE_ENDPOINTS | READ_ENDPOINTS


@app.after_request
def add_api_cors_headers(response):
    origin = request.headers.get("Origin")
    if not (
        request.path.startswith("/api/")
        or request.path.startswith("/screenshots/")
        or request.path == "/video_feed"
    ):
        return response

    vary_headers = {
        value.strip().lower()
        for value in response.headers.get("Vary", "").split(",")
    }
    if "origin" not in vary_headers:
        response.headers.add("Vary", "Origin")

    if origin != FRONTEND_ORIGIN:
        return response

    response.headers["Access-Control-Allow-Origin"] = FRONTEND_ORIGIN

    if request.method == "OPTIONS" and request.url_rule is not None:
        route_methods = request.url_rule.methods or set()
        allowed_methods = [
            method for method in ("GET", "POST", "PATCH")
            if method in route_methods
        ]
        if allowed_methods:
            response.headers["Access-Control-Allow-Methods"] = ", ".join(allowed_methods)

        if request.url_rule.endpoint in PROTECTED_ENDPOINTS:
            allowed_headers = ["X-API-Key"]
            if request.url_rule.endpoint in {"api_update_status", "api_manage_config"}:
                allowed_headers.insert(0, "Content-Type")
            response.headers["Access-Control-Allow-Headers"] = ", ".join(allowed_headers)
            response.headers["Access-Control-Max-Age"] = "600"

    return response


@app.before_request
def require_api_key():
    protected_write = request.method == "POST" and request.endpoint in WRITE_ENDPOINTS
    protected_read = request.method in {"GET", "HEAD"} and request.endpoint in READ_ENDPOINTS
    if request.method == "OPTIONS" or not (protected_write or protected_read):
        return None

    expected_token = os.environ.get("API_WRITE_TOKEN", "")
    if not expected_token:
        logger.error("Rejected protected request because API_WRITE_TOKEN is not configured")
        return jsonify({"error": "Data access and write operations are disabled until API_WRITE_TOKEN is configured."}), 503

    supplied_token = request.headers.get("X-API-Key", "")
    if not hmac.compare_digest(
        supplied_token.encode("utf-8"),
        expected_token.encode("utf-8"),
    ):
        return jsonify({"error": "A valid prototype API access key is required."}), 401

    return None


@app.errorhandler(HTTPException)
def handle_api_http_error(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": error.description}), error.code
    return error


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    if not request.path.startswith("/api/"):
        raise error

    logger.error(
        "Unhandled API error for %s %s",
        request.method,
        request.path,
        exc_info=(type(error), error, error.__traceback__),
    )
    return jsonify({"error": "Internal server error"}), 500


# Global state sharing class
class SharedState:
    current_frame = None          # Stores the annotated frame for MJPEG stream
    active_student_count = 0      # Number of currently tracked students
    active_alert_count = 0        # Number of active alerts in current frame
    student_risk_scores = {}      # Student ID -> risk score percentage
    monitoring_active = True      # Toggle flag
    audio_metrics = {}            # Stores RMS, volume_db, speech flag
    suspicious_engine_ref = None  # Reference to active SuspiciousEngine instance
    camera_ref = None             # Reference to active CameraManager instance


def _json_safe(value):
    """Convert NumPy scalar values in runtime telemetry to JSON types."""
    """Convert NumPy scalar/array values, dates, and enums to standard JSON types."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    return value

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
        
        event_data = _json_safe({
            "timestamp": time.time(),
            "active_students": SharedState.active_student_count,
            "active_alerts": SharedState.active_alert_count,
            "total_logged_alerts": db_stats["total_alerts"],
            "student_risk_scores": SharedState.student_risk_scores,
            "audio_metrics": SharedState.audio_metrics,
            "new_alert": newest_alert if (newest_alert and newest_alert["id"] > last_sent_alert_id) else None
        })
        
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

@app.route('/api/health')
def api_health():
    """Report application and database readiness without exposing filesystem paths."""
    try:
        database = get_manager().health_check()
    except (OSError, sqlite3.Error):
        logger.exception("Database health check failed")
        return jsonify({
            "status": "degraded",
            "application": "running",
            "database": {
                "status": "unavailable",
                "path_configured": True,
                "required_tables": {"incidents": False},
                "missing_required_columns": None,
                "schema_initialized": DATABASE_SCHEMA_INITIALIZED,
            },
        }), 503

    ready = database["connected"] and database["schema_ready"]
    return jsonify({
        "status": "ok" if ready else "degraded",
        "application": "running",
        "database": {
            "status": "connected",
            "path_configured": True,
            "required_tables": database["required_tables"],
            "missing_required_columns": database["missing_required_columns"],
            "schema_initialized": DATABASE_SCHEMA_INITIALIZED,
        },
    }), 200 if ready else 503

@app.route('/api/stats')
def api_stats():
    """
    Returns real-time system statistics, risk scores, and audio metrics.
    """
    db_stats = get_stats()
    return jsonify(_json_safe({
        "active_students": SharedState.active_student_count,
        "active_alerts": SharedState.active_alert_count,
        "total_logged_alerts": db_stats["total_alerts"],
        "unique_students_flagged": db_stats["unique_students"],
        "severity_counts": db_stats.get("severity_counts", {}),
        "status_counts": db_stats.get("status_counts", {}),
        "category_counts": db_stats.get("category_counts", {}),
        "student_risk_scores": SharedState.student_risk_scores,
        "audio_metrics": SharedState.audio_metrics
    }))

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
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "A JSON incident status object is required"}), 400
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
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "A JSON configuration object is required"}), 400

        numeric_updates = {}
        numeric_fields = {
            "yaw": "HEAD_YAW_THRESHOLD",
            "pitch": "HEAD_PITCH_THRESHOLD",
            "duration": "SUSPICIOUS_DURATION",
            "audio_threshold": "AUDIO_NOISE_THRESHOLD",
        }
        try:
            for key, attribute in numeric_fields.items():
                if key in data:
                    value = data[key]
                    if isinstance(value, bool):
                        raise ValueError
                    parsed = float(value)
                    if not math.isfinite(parsed):
                        raise ValueError
                    numeric_updates[attribute] = parsed
        except (TypeError, ValueError):
            return jsonify({"error": "Numeric configuration values are invalid"}), 400

        boolean_fields = {
            "enable_object_detection": "ENABLE_OBJECT_DETECTION",
            "enable_absence_detection": "ENABLE_ABSENCE_DETECTION",
            "enable_audio_detection": "ENABLE_AUDIO_DETECTION",
        }
        if any(
            key in data and not isinstance(data[key], bool)
            for key in boolean_fields
        ):
            return jsonify({"error": "Configuration feature flags must be booleans"}), 400

        for attribute, value in numeric_updates.items():
            setattr(cfg, attribute, value)
        for key, attribute in boolean_fields.items():
            if key in data:
                setattr(cfg, attribute, data[key])

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

@app.route('/api/validation/results')
def api_validation_results():
    """Return summaries from local validation result JSON files."""
    results_dir = Path(cfg.BASE_DIR) / "validation" / "results"
    summaries = []
    if results_dir.is_dir():
        for result_path in sorted(results_dir.rglob("*.json")):
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            summaries.append({
                "session": result.get("session", result_path.stem),
                "result_file": str(result_path.relative_to(cfg.BASE_DIR)),
                "duration_seconds": result.get("duration_seconds"),
                "frames_processed": result.get("frames_processed"),
                "average_fps": result.get("average_fps"),
                "evaluation": result.get("evaluation"),
            })
    return jsonify(summaries)

@app.route('/api/validation/latest')
def api_validation_latest():
    """Return the newest local validation result summary, if one exists."""
    results_dir = Path(cfg.BASE_DIR) / "validation" / "results"
    result_paths = sorted(
        results_dir.rglob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ) if results_dir.is_dir() else []
    for result_path in result_paths:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        return jsonify(result)
    return jsonify({"status": "no_results"}), 404

@app.route('/screenshots/<path:filename>')
def get_screenshot(filename):
    """
    Serves saved screenshots.
    """
    return send_from_directory(cfg.SCREENSHOT_DIR, filename)

@app.route('/api/control/<action>', methods=['POST'])
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
    return jsonify({"error": "Unknown monitoring action"}), 404

_flask_server = None
_flask_socket = None

def run_flask_server():
    """
    Starts the Flask app on the configured port.
    """
    global _flask_server, _flask_socket
    import logging
    import socket
    import time
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    from werkzeug.serving import make_server

    # Attempt to bind socket with SO_REUSEADDR, retrying if port was recently used
    server_sock = None
    for attempt in range(10):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except (AttributeError, OSError):
                pass
            s.bind((cfg.FLASK_HOST, cfg.FLASK_PORT))
            s.listen(128)
            server_sock = s
            break
        except OSError as e:
            s.close()
            if attempt < 9:
                time.sleep(0.5)
            else:
                print(f"[DASHBOARD] Warning: Could not bind {cfg.FLASK_HOST}:{cfg.FLASK_PORT}: {e}")
                return

    _flask_socket = server_sock
    try:
        _flask_server = make_server(
            cfg.FLASK_HOST, cfg.FLASK_PORT, app, threaded=True, fd=server_sock.fileno()
        )
        _flask_server.serve_forever()
    except Exception as e:
        print(f"[DASHBOARD] Server error: {e}")
    finally:
        if _flask_server is not None:
            try:
                _flask_server.server_close()
            except Exception:
                pass
        if _flask_socket is not None:
            try:
                _flask_socket.close()
            except Exception:
                pass
            _flask_socket = None

def stop_flask_server():
    """
    Stops the running Flask server cleanly.
    """
    global _flask_server, _flask_socket
    if _flask_server is not None:
        try:
            _flask_server.shutdown()
        except Exception:
            pass
        try:
            _flask_server.server_close()
        except Exception:
            pass
        _flask_server = None
    if _flask_socket is not None:
        try:
            _flask_socket.close()
        except Exception:
            pass
        _flask_socket = None
