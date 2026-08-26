"""
Tests for Event Manager
-----------------------
Tests for EventManager.handle_event() with mock dependencies.

Run:
    python -m pytest tests/test_event_manager.py -v
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.event_manager import EventManager


class TestEventManagerInit(unittest.TestCase):

    def test_init_with_no_dependencies(self):
        em = EventManager()
        self.assertIsNone(em.db_manager)
        self.assertIsNone(em.timeline)
        self.assertIsNone(em.notifier)
        self.assertIsNone(em.logger)

    def test_init_with_all_dependencies(self):
        db = MagicMock()
        tl = MagicMock()
        notifier = MagicMock()
        logger = MagicMock()
        em = EventManager(db_manager=db, timeline=tl, notifier=notifier, logger=logger)
        self.assertIs(em.db_manager, db)
        self.assertIs(em.timeline, tl)
        self.assertIs(em.notifier, notifier)
        self.assertIs(em.logger, logger)


class TestHandleEventReturnsIncident(unittest.TestCase):

    def setUp(self):
        self.em = EventManager()

    def test_returns_dict(self):
        result = self.em.handle_event(1, "CELL_PHONE", 60)
        self.assertIsInstance(result, dict)

    def test_incident_contains_required_fields(self):
        result = self.em.handle_event(1, "CELL_PHONE", 60)
        self.assertIn("student_id", result)
        self.assertIn("event", result)
        self.assertIn("risk_score", result)
        self.assertIn("timestamp", result)
        self.assertIn("screenshot", result)

    def test_incident_values_match_input(self):
        result = self.em.handle_event(5, "STANDING", 35)
        self.assertEqual(result["student_id"], 5)
        self.assertEqual(result["event"], "STANDING")
        self.assertEqual(result["risk_score"], 35)

    def test_screenshot_none_by_default(self):
        result = self.em.handle_event(1, "LOOK_LEFT", 8)
        self.assertIsNone(result["screenshot"])

    def test_screenshot_path_stored(self):
        result = self.em.handle_event(1, "CELL_PHONE", 60, screenshot_path="/tmp/shot.jpg")
        self.assertEqual(result["screenshot"], "/tmp/shot.jpg")

    def test_timestamp_format(self):
        import re
        result = self.em.handle_event(1, "TALKING", 20)
        pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
        self.assertRegex(result["timestamp"], pattern)


class TestHandleEventCallsDependencies(unittest.TestCase):

    def test_timeline_add_event_called(self):
        timeline = MagicMock()
        em = EventManager(timeline=timeline)
        em.handle_event(1, "LOOK_LEFT", 8)
        timeline.add_event.assert_called_once()
        call_args = timeline.add_event.call_args[0]
        self.assertEqual(call_args[0], 1)

    def test_db_manager_insert_incident_called(self):
        db = MagicMock()
        em = EventManager(db_manager=db)
        em.handle_event(2, "PHONE", 60)
        db.insert_incident.assert_called_once()

    def test_notifier_send_alert_called(self):
        notifier = MagicMock()
        em = EventManager(notifier=notifier)
        em.handle_event(3, "STANDING", 25)
        notifier.send_alert.assert_called_once()

    def test_logger_log_called(self):
        logger = MagicMock()
        em = EventManager(logger=logger)
        em.handle_event(4, "LEANING", 10)
        logger.log.assert_called_once()

    def test_no_calls_when_dependencies_are_none(self):
        # Should not raise any errors with all None dependencies
        em = EventManager()
        try:
            em.handle_event(1, "NORMAL", 0)
        except Exception as e:
            self.fail(f"handle_event raised unexpected exception: {e}")

    def test_all_dependencies_called_together(self):
        db = MagicMock()
        timeline = MagicMock()
        notifier = MagicMock()
        logger = MagicMock()
        em = EventManager(db_manager=db, timeline=timeline, notifier=notifier, logger=logger)
        em.handle_event(1, "CELL_PHONE", 60)
        db.insert_incident.assert_called_once()
        timeline.add_event.assert_called_once()
        notifier.send_alert.assert_called_once()
        logger.log.assert_called_once()

    def test_incident_passed_to_db_is_correct(self):
        db = MagicMock()
        em = EventManager(db_manager=db)
        em.handle_event(7, "BOOK", 30)
        incident = db.insert_incident.call_args[0][0]
        self.assertEqual(incident["student_id"], 7)
        self.assertEqual(incident["event"], "BOOK")
        self.assertEqual(incident["risk_score"], 30)


class TestHandleEventEdgeCases(unittest.TestCase):

    def setUp(self):
        self.em = EventManager()

    def test_zero_risk_score(self):
        result = self.em.handle_event(1, "NORMAL", 0)
        self.assertEqual(result["risk_score"], 0)

    def test_max_risk_score(self):
        result = self.em.handle_event(1, "MULTIPLE_PERSON", 100)
        self.assertEqual(result["risk_score"], 100)

    def test_different_student_ids(self):
        for sid in [1, 2, 3, 100]:
            result = self.em.handle_event(sid, "LOOK_LEFT", 8)
            self.assertEqual(result["student_id"], sid)


if __name__ == "__main__":
    unittest.main()
