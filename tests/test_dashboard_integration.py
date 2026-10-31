"""End-to-end Flask dashboard tests using a temporary SQLite database."""

import os
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import database.db_manager as db_manager
import config.config as config
from dashboard.app import SharedState, app


class TestDashboardIntegration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.database_path = os.path.join(
            self.temp_dir,
            "exam_monitoring.db"
        )
        self.reports_dir = os.path.join(self.temp_dir, "reports")
        self.original_config = {
            "HEAD_YAW_THRESHOLD": config.HEAD_YAW_THRESHOLD,
            "HEAD_PITCH_THRESHOLD": config.HEAD_PITCH_THRESHOLD,
            "SUSPICIOUS_DURATION": config.SUSPICIOUS_DURATION,
            "AUDIO_NOISE_THRESHOLD": config.AUDIO_NOISE_THRESHOLD,
            "ENABLE_OBJECT_DETECTION": config.ENABLE_OBJECT_DETECTION,
            "ENABLE_ABSENCE_DETECTION": config.ENABLE_ABSENCE_DETECTION,
            "ENABLE_AUDIO_DETECTION": config.ENABLE_AUDIO_DETECTION,
        }
        self.db_patch = patch.object(
            db_manager,
            "DB_PATH",
            self.database_path
        )
        self.reports_patch = patch(
            "dashboard.app.cfg.REPORTS_DIR",
            self.reports_dir
        )
        self.db_patch.start()
        self.reports_patch.start()
        self.api_key_patch = patch.dict(
            os.environ,
            {"API_WRITE_TOKEN": "test-operator-key"}
        )
        self.api_key_patch.start()
        db_manager.init_db()
        self.client = app.test_client()
        self.client.environ_base["HTTP_X_API_KEY"] = "test-operator-key"
        SharedState.current_frame = None
        SharedState.monitoring_active = True
        SharedState.active_student_count = 0
        SharedState.active_alert_count = 0
        SharedState.student_risk_scores = {}
        SharedState.audio_metrics = {}

    def tearDown(self):
        SharedState.current_frame = None
        self.reports_patch.stop()
        self.db_patch.stop()
        self.api_key_patch.stop()
        config.HEAD_YAW_THRESHOLD = self.original_config[
            "HEAD_YAW_THRESHOLD"
        ]
        config.HEAD_PITCH_THRESHOLD = self.original_config[
            "HEAD_PITCH_THRESHOLD"
        ]
        config.SUSPICIOUS_DURATION = self.original_config[
            "SUSPICIOUS_DURATION"
        ]
        config.AUDIO_NOISE_THRESHOLD = self.original_config[
            "AUDIO_NOISE_THRESHOLD"
        ]
        config.ENABLE_OBJECT_DETECTION = self.original_config[
            "ENABLE_OBJECT_DETECTION"
        ]
        config.ENABLE_ABSENCE_DETECTION = self.original_config[
            "ENABLE_ABSENCE_DETECTION"
        ]
        config.ENABLE_AUDIO_DETECTION = self.original_config[
            "ENABLE_AUDIO_DETECTION"
        ]
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_headers(self, json_body=False):
        headers = {"X-API-Key": "test-operator-key"}
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _seed_incidents(self):
        incidents = [
            (1, "LOOK_LEFT", "HIGH", 35.0),
            (1, "CELL_PHONE", "CRITICAL", 95.0),
            (1, "HAND_FACE", "MEDIUM", 45.0),
            (2, "MULTIPLE_PERSON", "HIGH", 80.0),
            (2, "STANDING", "LOW", 25.0),
        ]
        return [
            db_manager.log_incident(
                student_id=student_id,
                activity=activity,
                severity=severity,
                risk_score=risk_score
            )
            for student_id, activity, severity, risk_score in incidents
        ]

    def test_alerts_filters_status_csv_and_report(self):
        incident_ids = self._seed_incidents()

        response = self.client.get("/api/alerts")
        self.assertEqual(response.status_code, 200)
        alerts = response.get_json()
        self.assertEqual(len(alerts), 5)
        self.assertEqual(alerts[0]["risk_score"], 25.0)

        student_alerts = self.client.get("/api/alerts?student_id=1")
        self.assertEqual(len(student_alerts.get_json()), 3)

        gaze_alerts = self.client.get("/api/alerts?category=GAZE")
        self.assertEqual(
            [item["activity"] for item in gaze_alerts.get_json()],
            ["LOOK_LEFT"]
        )

        critical_alerts = self.client.get("/api/alerts?severity=CRITICAL")
        self.assertEqual(
            [item["activity"] for item in critical_alerts.get_json()],
            ["CELL_PHONE"]
        )

        status_response = self.client.post(
            f"/api/incidents/{incident_ids[0]}/status",
            json={"status": "CONFIRMED"},
            headers=self._write_headers()
        )
        self.assertEqual(status_response.status_code, 200)
        confirmed = self.client.get("/api/alerts?status=CONFIRMED")
        self.assertEqual(len(confirmed.get_json()), 1)
        self.assertEqual(confirmed.get_json()[0]["activity"], "LOOK_LEFT")

        csv_response = self.client.get("/api/incidents/export")
        self.assertEqual(csv_response.status_code, 200)
        self.assertIn("CELL_PHONE", csv_response.get_data(as_text=True))
        self.assertIn("95.0", csv_response.get_data(as_text=True))

        report_response = self.client.get(
            "/api/report/pdf?exam_name=Integration%20Exam"
        )
        self.assertEqual(report_response.status_code, 200)
        self.assertIn("Integration Exam", report_response.get_data(as_text=True))
        self.assertTrue(os.listdir(self.reports_dir))

    def test_dashboard_streams_and_handles_empty_or_missing_data(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/api/alerts").get_json(), [])

        missing_evidence = self.client.get("/screenshots/missing.jpg")
        self.assertEqual(missing_evidence.status_code, 404)

        SharedState.current_frame = np.zeros(
            (32, 32, 3),
            dtype=np.uint8
        )
        video_response = self.client.get(
            "/video_feed",
            buffered=False
        )
        video_chunk = next(video_response.response)
        self.assertIn(b"Content-Type: image/jpeg", video_chunk)
        video_response.close()

        SharedState.student_risk_scores = {1: 95.0}
        SharedState.audio_metrics = {
            "speech_detected": np.bool_(False),
            "rms": np.float32(0.08),
        }
        SharedState.active_student_count = 1
        stats_response = self.client.get("/api/stats")
        self.assertEqual(stats_response.status_code, 200)
        self.assertFalse(stats_response.get_json()["audio_metrics"]["speech_detected"])

        event_response = self.client.get(
            "/api/stream-events",
            buffered=False
        )
        event_chunk = next(event_response.response).decode()
        self.assertIn('"active_students": 1', event_chunk)
        event_response.close()

    def test_sensitive_reads_require_the_temporary_api_access_key(self):
        unauthenticated = app.test_client()
        protected_paths = [
            "/api/stats",
            "/api/alerts",
            "/api/stream-events",
            "/api/incidents/export",
            "/api/report/pdf",
            "/api/config",
            "/api/validation/results",
            "/api/validation/latest",
            "/screenshots/missing.jpg",
            "/video_feed",
        ]
        for path in protected_paths:
            with self.subTest(path=path):
                response = unauthenticated.get(path, buffered=False)
                self.assertEqual(response.status_code, 401)
                if path == "/api/stream-events":
                    response.close()

        rejected = unauthenticated.get(
            "/api/stats",
            headers={"X-API-Key": "incorrect"}
        )
        self.assertEqual(rejected.status_code, 401)
        self.assertEqual(unauthenticated.head("/api/alerts").status_code, 401)

        health = unauthenticated.get("/api/health")
        self.assertEqual(health.status_code, 200)

    def test_protected_read_fails_closed_when_api_key_is_unconfigured(self):
        with patch.dict(os.environ, {"API_WRITE_TOKEN": ""}):
            response = app.test_client().get("/api/alerts")

        self.assertEqual(response.status_code, 503)
        self.assertIn("disabled", response.get_json()["error"])

    def test_api_cors_is_allowlisted_and_preflight_matches_route(self):
        frontend_origin = "https://holistic-detection.vercel.app"

        allowed = self.client.get(
            "/api/alerts",
            headers={"Origin": frontend_origin}
        )
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(
            allowed.headers["Access-Control-Allow-Origin"],
            frontend_origin
        )
        self.assertIn("Origin", allowed.headers["Vary"])
        self.assertNotIn("Access-Control-Allow-Credentials", allowed.headers)

        stats = self.client.get(
            "/api/stats",
            headers={"Origin": frontend_origin}
        )
        self.assertEqual(stats.status_code, 200)
        self.assertEqual(stats.get_json()["total_logged_alerts"], 0)
        self.assertEqual(
            stats.headers["Access-Control-Allow-Origin"],
            frontend_origin
        )

        csv_export = self.client.get(
            "/api/incidents/export",
            headers={"Origin": frontend_origin}
        )
        self.assertEqual(csv_export.status_code, 200)
        self.assertEqual(
            csv_export.headers["Access-Control-Allow-Origin"],
            frontend_origin
        )

        report = self.client.get(
            "/api/report/pdf",
            headers={"Origin": frontend_origin}
        )
        self.assertEqual(report.status_code, 200)
        self.assertEqual(
            report.headers["Access-Control-Allow-Origin"],
            frontend_origin
        )

        event_stream = self.client.get(
            "/api/stream-events",
            headers={"Origin": frontend_origin},
            buffered=False
        )
        self.assertEqual(event_stream.status_code, 200)
        self.assertEqual(
            event_stream.headers["Access-Control-Allow-Origin"],
            frontend_origin
        )
        event_stream.close()

        disallowed = self.client.get(
            "/api/alerts",
            headers={"Origin": "https://untrusted.example"}
        )
        self.assertNotIn("Access-Control-Allow-Origin", disallowed.headers)

        preflight = self.client.options(
            "/api/incidents/1/status",
            headers={
                "Origin": frontend_origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            }
        )
        self.assertEqual(preflight.status_code, 200)
        self.assertEqual(
            preflight.headers["Access-Control-Allow-Origin"],
            frontend_origin
        )
        self.assertEqual(
            preflight.headers["Access-Control-Allow-Methods"],
            "POST"
        )
        self.assertEqual(
            preflight.headers["Access-Control-Allow-Headers"],
            "Content-Type, X-API-Key"
        )

        config_preflight = self.client.options(
            "/api/config",
            headers={
                "Origin": frontend_origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            }
        )
        self.assertEqual(
            config_preflight.headers["Access-Control-Allow-Methods"],
            "GET, POST"
        )
        self.assertNotIn(
            "PATCH",
            config_preflight.headers["Access-Control-Allow-Methods"]
        )
        self.assertEqual(
            config_preflight.headers["Access-Control-Allow-Headers"],
            "Content-Type, X-API-Key"
        )

        control_preflight = self.client.options(
            "/api/control/pause",
            headers={
                "Origin": frontend_origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "x-api-key",
            }
        )
        self.assertEqual(
            control_preflight.headers["Access-Control-Allow-Methods"],
            "POST"
        )
        self.assertEqual(
            control_preflight.headers["Access-Control-Allow-Headers"],
            "X-API-Key"
        )

        screenshot_preflight = self.client.options(
            "/screenshots/evidence.jpg",
            headers={
                "Origin": frontend_origin,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "x-api-key",
            }
        )
        self.assertEqual(
            screenshot_preflight.headers["Access-Control-Allow-Origin"],
            frontend_origin
        )
        self.assertEqual(
            screenshot_preflight.headers["Access-Control-Allow-Methods"],
            "GET"
        )
        self.assertEqual(
            screenshot_preflight.headers["Access-Control-Allow-Headers"],
            "X-API-Key"
        )

    def test_api_database_failure_is_logged_json_and_cors_enabled(self):
        broken_database_path = os.path.join(self.temp_dir, "missing-schema.db")
        with patch.object(db_manager, "DB_PATH", broken_database_path):
            response = self.client.get(
                "/api/stats",
                headers={"Origin": "https://holistic-detection.vercel.app"}
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.get_json(),
            {"error": "Internal server error"}
        )
        self.assertEqual(
            response.headers["Access-Control-Allow-Origin"],
            "https://holistic-detection.vercel.app"
        )
        self.assertNotIn("sqlite3", response.get_data(as_text=True))

    def test_legacy_schema_migrates_without_losing_incidents(self):
        legacy_path = os.path.join(self.temp_dir, "legacy.db")
        connection = sqlite3.connect(legacy_path)
        connection.execute("""
            CREATE TABLE incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                activity TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        connection.execute(
            """
            INSERT INTO incidents (student_id, activity, timestamp)
            VALUES (?, ?, ?)
            """,
            (17, "LOOK_LEFT", "2026-10-31 10:00:00")
        )
        connection.commit()
        connection.close()

        with patch.object(db_manager, "DB_PATH", legacy_path):
            db_manager.init_db()
            stats = db_manager.get_stats()
            incidents = db_manager.get_all_incidents()

        self.assertEqual(stats["total_alerts"], 1)
        self.assertEqual(stats["unique_students"], 1)
        self.assertEqual(incidents[0]["student_id"], 17)
        self.assertEqual(incidents[0]["activity"], "LOOK_LEFT")
        self.assertEqual(incidents[0]["severity"], "MEDIUM")
        self.assertIsNone(incidents[0]["screenshot_path"])

    def test_health_reports_database_readiness_without_exposing_path(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["application"], "running")
        self.assertEqual(payload["database"]["status"], "connected")
        self.assertTrue(payload["database"]["schema_initialized"])
        self.assertTrue(payload["database"]["required_tables"]["incidents"])
        self.assertEqual(payload["database"]["missing_required_columns"], [])
        self.assertNotIn(self.temp_dir, response.get_data(as_text=True))

    def test_health_reports_missing_schema_and_database_failure(self):
        uninitialized_path = os.path.join(self.temp_dir, "uninitialized.db")
        with patch.object(db_manager, "DB_PATH", uninitialized_path):
            missing_schema = self.client.get("/api/health")
        self.assertEqual(missing_schema.status_code, 503)
        self.assertEqual(
            missing_schema.get_json()["database"]["required_tables"]["incidents"],
            False
        )
        self.assertGreater(
            len(missing_schema.get_json()["database"]["missing_required_columns"]),
            0
        )

        with patch.object(db_manager, "DB_PATH", self.temp_dir):
            unavailable = self.client.get("/api/health")
        self.assertEqual(unavailable.status_code, 503)
        self.assertEqual(
            unavailable.get_json()["database"]["status"],
            "unavailable"
        )
        self.assertNotIn(self.temp_dir, unavailable.get_data(as_text=True))

    def test_wsgi_import_initializes_a_fresh_database(self):
        database_path = os.path.join(self.temp_dir, "wsgi-fresh.db")
        environment = os.environ.copy()
        environment["DB_PATH"] = database_path
        environment["API_WRITE_TOKEN"] = "test-operator-key"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os; "
                    "from config.config import DB_PATH; "
                    "from database.db_manager import get_manager; "
                    "assert os.path.abspath(DB_PATH) == get_manager().db_path; "
                    "from dashboard.app import app; "
                    "client = app.test_client(); "
                    "headers = {'X-API-Key': os.environ['API_WRITE_TOKEN']}; "
                    "print(client.get('/api/health').status_code); "
                    "print(client.get('/api/stats', headers=headers).status_code); "
                    "print(client.get('/api/alerts', headers=headers).status_code)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(result.stdout.strip().splitlines(), ["200", "200", "200"])

    def test_write_routes_require_operator_key_and_fail_closed_when_unconfigured(self):
        incident_id = self._seed_incidents()[0]
        path = f"/api/incidents/{incident_id}/status"

        missing_key = self.client.post(
            path,
            json={"status": "CONFIRMED"},
            headers={"X-API-Key": ""}
        )
        self.assertEqual(missing_key.status_code, 401)

        invalid_key = self.client.post(
            path,
            json={"status": "CONFIRMED"},
            headers={"X-API-Key": "wrong-key"}
        )
        self.assertEqual(invalid_key.status_code, 401)

        self.assertEqual(
            self.client.post(
                "/api/incidents/clear",
                headers={"X-API-Key": ""}
            ).status_code,
            401
        )
        self.assertEqual(
            self.client.post(
                "/api/config",
                json={},
                headers={"X-API-Key": ""}
            ).status_code,
            401
        )
        self.assertEqual(
            self.client.post(
                "/api/control/pause",
                headers={"X-API-Key": ""}
            ).status_code,
            401
        )
        legacy_get = self.client.get("/api/control/pause")
        self.assertEqual(legacy_get.status_code, 405)
        self.assertIsInstance(legacy_get.get_json()["error"], str)

        with patch.dict(os.environ, {"API_WRITE_TOKEN": ""}):
            unconfigured = self.client.post(
                path,
                json={"status": "CONFIRMED"},
                headers=self._write_headers()
            )
        self.assertEqual(unconfigured.status_code, 503)
        self.assertEqual(
            self.client.get("/api/alerts?status=CONFIRMED").get_json(),
            []
        )

    def test_status_config_control_and_invalid_inputs(self):
        incident_id = self._seed_incidents()[0]

        self.assertEqual(
            self.client.post(
                f"/api/incidents/{incident_id}/status",
                json={"status": "INVALID"},
                headers=self._write_headers()
            ).status_code,
            404
        )
        self.assertEqual(
            self.client.post(
                f"/api/incidents/{incident_id}/status",
                json={},
                headers=self._write_headers()
            ).status_code,
            400
        )
        self.assertEqual(
            self.client.get("/api/alerts?student_id=not-a-number").status_code,
            200
        )

        self.assertEqual(
            self.client.post(
                "/api/control/pause",
                headers=self._write_headers()
            ).get_json()["status"],
            "paused"
        )
        self.assertEqual(
            self.client.post(
                "/api/control/resume",
                headers=self._write_headers()
            ).get_json()["status"],
            "running"
        )
        self.assertEqual(
            self.client.post(
                "/api/control/unknown",
                headers=self._write_headers()
            ).status_code,
            404
        )

        config_response = self.client.post(
            "/api/config",
            json={"yaw": 30, "enable_object_detection": False},
            headers=self._write_headers()
        )
        self.assertEqual(config_response.status_code, 200)
        self.assertEqual(config_response.get_json()["yaw_threshold"], 30.0)

        invalid_config = self.client.post(
            "/api/config",
            json={"yaw": "not-a-number"},
            headers=self._write_headers()
        )
        self.assertEqual(invalid_config.status_code, 400)

        invalid_flags = self.client.post(
            "/api/config",
            json={"yaw": 40, "enable_audio_detection": "false"},
            headers=self._write_headers()
        )
        self.assertEqual(invalid_flags.status_code, 400)
        self.assertEqual(
            self.client.get("/api/config").get_json()["yaw_threshold"],
            30.0
        )

    def test_validation_results_endpoint_handles_empty_directory(self):
        response = self.client.get("/api/validation/results")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), [])
        latest = self.client.get("/api/validation/latest")
        self.assertEqual(latest.status_code, 404)

    def test_validation_results_endpoint_reads_saved_summary(self):
        results_dir = os.path.join(self.temp_dir, "validation", "results")
        os.makedirs(results_dir)
        result_path = os.path.join(results_dir, "session.json")
        with open(result_path, "w", encoding="utf-8") as result_file:
            json.dump({
                "session": "session",
                "duration_seconds": 4.0,
                "frames_processed": 20,
                "average_fps": 5.0,
                "evaluation": {"overall": {"f1": 1.0}},
            }, result_file)

        with patch("dashboard.app.cfg.BASE_DIR", self.temp_dir):
            response = self.client.get("/api/validation/results")
            latest = self.client.get("/api/validation/latest")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()[0]["session"], "session")
        self.assertEqual(response.get_json()[0]["average_fps"], 5.0)
        self.assertEqual(latest.status_code, 200)
        self.assertEqual(latest.get_json()["session"], "session")


if __name__ == "__main__":
    unittest.main()
