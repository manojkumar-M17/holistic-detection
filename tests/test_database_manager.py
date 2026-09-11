"""
Database Manager Tests
----------------------
Tests SQLite database operations for the
AI Exam Hall Monitoring System.

Run:
    python -m pytest tests/test_database.py -v
"""

import sys
import os
import unittest
import tempfile
import shutil
import sqlite3
from unittest.mock import patch

# Add project root to Python path
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import database.db_manager as db_manager


class TestDatabaseManager(unittest.TestCase):

    def setUp(self):

        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()

        # Temporary database
        self.test_db = os.path.join(
            self.temp_dir,
            "test_exam_monitoring.db"
        )

        # Patch DB_PATH
        self.db_patch = patch.object(
            db_manager,
            "DB_PATH",
            self.test_db
        )

        self.db_patch.start()

        # Initialize database
        db_manager.init_db()


    def tearDown(self):

        self.db_patch.stop()

        shutil.rmtree(
            self.temp_dir,
            ignore_errors=True
        )


    # ==========================================
    # DATABASE INITIALIZATION TESTS
    # ==========================================

    def test_database_file_created(self):

        self.assertTrue(
            os.path.exists(
                self.test_db
            )
        )


    def test_incidents_table_created(self):

        conn = sqlite3.connect(
            self.test_db
        )

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            AND name='incidents'
            """
        )

        result = cursor.fetchone()

        conn.close()

        self.assertIsNotNone(
            result
        )


    # ==========================================
    # LOG INCIDENT TESTS
    # ==========================================

    def test_log_incident_returns_id(self):

        incident_id = db_manager.log_incident(
            student_id=1,
            activity="CELL_PHONE",
            screenshot_path="screenshots/test.jpg"
        )

        self.assertIsInstance(
            incident_id,
            int
        )

        self.assertGreater(
            incident_id,
            0
        )


    def test_logged_incident_can_be_retrieved(self):

        db_manager.log_incident(
            student_id=1,
            activity="LOOK_LEFT",
            screenshot_path="screenshots/look.jpg",
            severity="LOW",
            category="GAZE",
            risk_score=10
        )

        incidents = db_manager.get_all_incidents()

        self.assertEqual(
            len(incidents),
            1
        )

        self.assertEqual(
            incidents[0]["student_id"],
            1
        )

        self.assertEqual(
            incidents[0]["activity"],
            "LOOK_LEFT"
        )


    def test_incident_severity_saved(self):

        db_manager.log_incident(
            student_id=2,
            activity="STANDING",
            screenshot_path="standing.jpg",
            severity="HIGH"
        )

        incidents = db_manager.get_all_incidents()

        self.assertEqual(
            incidents[0]["severity"],
            "HIGH"
        )


    def test_incident_category_saved(self):

        db_manager.log_incident(
            student_id=3,
            activity="CELL_PHONE",
            screenshot_path="phone.jpg",
            category="FORBIDDEN_OBJECT"
        )

        incidents = db_manager.get_all_incidents()

        self.assertEqual(
            incidents[0]["category"],
            "FORBIDDEN_OBJECT"
        )


    def test_incident_risk_score_saved(self):

        db_manager.log_incident(
            student_id=4,
            activity="TALKING",
            screenshot_path="talking.jpg",
            risk_score=45.5
        )

        incidents = db_manager.get_all_incidents()

        self.assertEqual(
            incidents[0]["risk_score"],
            45.5
        )


    def test_default_status_is_unreviewed(self):

        db_manager.log_incident(
            student_id=1,
            activity="LOOK_RIGHT",
            screenshot_path="right.jpg"
        )

        incidents = db_manager.get_all_incidents()

        self.assertEqual(
            incidents[0]["status"],
            "UNREVIEWED"
        )


    # ==========================================
    # UPDATE STATUS TESTS
    # ==========================================

    def test_update_incident_status_confirmed(self):

        incident_id = db_manager.log_incident(
            student_id=1,
            activity="CELL_PHONE",
            screenshot_path="phone.jpg"
        )

        result = db_manager.update_incident_status(
            incident_id,
            "CONFIRMED"
        )

        self.assertTrue(
            result
        )

        incidents = db_manager.get_all_incidents()

        self.assertEqual(
            incidents[0]["status"],
            "CONFIRMED"
        )


    def test_update_incident_status_dismissed(self):

        incident_id = db_manager.log_incident(
            student_id=1,
            activity="LOOK_LEFT",
            screenshot_path="look.jpg"
        )

        result = db_manager.update_incident_status(
            incident_id,
            "DISMISSED"
        )

        self.assertTrue(
            result
        )


    def test_invalid_status_returns_false(self):

        incident_id = db_manager.log_incident(
            student_id=1,
            activity="TEST",
            screenshot_path="test.jpg"
        )

        result = db_manager.update_incident_status(
            incident_id,
            "INVALID"
        )

        self.assertFalse(
            result
        )


    # ==========================================
    # FILTER TESTS
    # ==========================================

    def test_filter_by_student_id(self):

        db_manager.log_incident(
            1,
            "LOOK_LEFT",
            "a.jpg"
        )

        db_manager.log_incident(
            2,
            "CELL_PHONE",
            "b.jpg"
        )

        incidents = db_manager.get_all_incidents(
            student_id=1
        )

        self.assertEqual(
            len(incidents),
            1
        )

        self.assertEqual(
            incidents[0]["student_id"],
            1
        )


    def test_filter_by_severity(self):

        db_manager.log_incident(
            1,
            "LOOK_LEFT",
            "a.jpg",
            severity="LOW"
        )

        db_manager.log_incident(
            2,
            "CELL_PHONE",
            "b.jpg",
            severity="CRITICAL"
        )

        incidents = db_manager.get_all_incidents(
            severity="CRITICAL"
        )

        self.assertEqual(
            len(incidents),
            1
        )

        self.assertEqual(
            incidents[0]["severity"],
            "CRITICAL"
        )


    def test_filter_by_category(self):

        db_manager.log_incident(
            1,
            "LOOK_LEFT",
            "a.jpg",
            category="GAZE"
        )

        db_manager.log_incident(
            2,
            "CELL_PHONE",
            "b.jpg",
            category="OBJECT"
        )

        incidents = db_manager.get_all_incidents(
            category="OBJECT"
        )

        self.assertEqual(
            len(incidents),
            1
        )

        self.assertEqual(
            incidents[0]["category"],
            "OBJECT"
        )


    def test_filter_by_status(self):

        incident_id = db_manager.log_incident(
            1,
            "CELL_PHONE",
            "a.jpg"
        )

        db_manager.update_incident_status(
            incident_id,
            "CONFIRMED"
        )

        incidents = db_manager.get_all_incidents(
            status="CONFIRMED"
        )

        self.assertEqual(
            len(incidents),
            1
        )


    # ==========================================
    # STATISTICS TESTS
    # ==========================================

    def test_get_stats_returns_dictionary(self):

        stats = db_manager.get_stats()

        self.assertIsInstance(
            stats,
            dict
        )


    def test_stats_total_alerts(self):

        db_manager.log_incident(
            1,
            "LOOK_LEFT",
            "a.jpg"
        )

        db_manager.log_incident(
            2,
            "CELL_PHONE",
            "b.jpg"
        )

        stats = db_manager.get_stats()

        self.assertEqual(
            stats["total_alerts"],
            2
        )


    def test_stats_unique_students(self):

        db_manager.log_incident(
            1,
            "LOOK_LEFT",
            "a.jpg"
        )

        db_manager.log_incident(
            1,
            "LOOK_RIGHT",
            "b.jpg"
        )

        db_manager.log_incident(
            2,
            "CELL_PHONE",
            "c.jpg"
        )

        stats = db_manager.get_stats()

        self.assertEqual(
            stats["unique_students"],
            2
        )


    # ==========================================
    # CSV EXPORT TEST
    # ==========================================

    def test_export_csv(self):

        db_manager.log_incident(
            1,
            "CELL_PHONE",
            "phone.jpg",
            severity="CRITICAL",
            category="OBJECT",
            risk_score=90
        )

        csv_data = db_manager.export_incidents_csv()

        self.assertIsInstance(
            csv_data,
            str
        )

        self.assertIn(
            "Incident ID",
            csv_data
        )

        self.assertIn(
            "CELL_PHONE",
            csv_data
        )


    # ==========================================
    # CLEAR DATABASE TEST
    # ==========================================

    def test_clear_all_incidents(self):

        db_manager.log_incident(
            1,
            "LOOK_LEFT",
            "a.jpg"
        )

        db_manager.log_incident(
            2,
            "CELL_PHONE",
            "b.jpg"
        )

        db_manager.clear_all_incidents()

        incidents = db_manager.get_all_incidents()

        self.assertEqual(
            len(incidents),
            0
        )


if __name__ == "__main__":

    unittest.main()