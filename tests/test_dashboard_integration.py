"""End-to-end Flask dashboard tests using a temporary SQLite database."""

import os
import shutil
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
            "ENABLE_OBJECT_DETECTION": config.ENABLE_OBJECT_DETECTION,
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
        db_manager.init_db()
        self.client = app.test_client()
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
        config.HEAD_YAW_THRESHOLD = self.original_config[
            "HEAD_YAW_THRESHOLD"
        ]
        config.ENABLE_OBJECT_DETECTION = self.original_config[
            "ENABLE_OBJECT_DETECTION"
        ]
        shutil.rmtree(self.temp_dir, ignore_errors=True)

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
            json={"status": "CONFIRMED"}
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

    def test_status_config_control_and_invalid_inputs(self):
        incident_id = self._seed_incidents()[0]

        self.assertEqual(
            self.client.post(
                f"/api/incidents/{incident_id}/status",
                json={"status": "INVALID"}
            ).status_code,
            404
        )
        self.assertEqual(
            self.client.post(
                f"/api/incidents/{incident_id}/status",
                json={}
            ).status_code,
            400
        )
        self.assertEqual(
            self.client.get("/api/alerts?student_id=not-a-number").status_code,
            200
        )

        self.assertEqual(
            self.client.get("/api/control/pause").get_json()["status"],
            "paused"
        )
        self.assertEqual(
            self.client.get("/api/control/resume").get_json()["status"],
            "running"
        )
        self.assertEqual(
            self.client.get("/api/control/unknown").get_json()["status"],
            "unknown"
        )

        config_response = self.client.post(
            "/api/config",
            json={"yaw": 30, "enable_object_detection": False}
        )
        self.assertEqual(config_response.status_code, 200)
        self.assertEqual(config_response.get_json()["yaw_threshold"], 30.0)

        invalid_config = self.client.post(
            "/api/config",
            json={"yaw": "not-a-number"}
        )
        self.assertEqual(invalid_config.status_code, 400)


if __name__ == "__main__":
    unittest.main()
