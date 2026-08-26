"""
Tests for Risk Engine
---------------------
Tests for RiskEngine and StudentRisk dataclass.

Run:
    python -m pytest tests/test_risk_engine.py -v
"""

import sys
import os
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.risk_engine import RiskEngine, StudentRisk


class TestStudentRisk(unittest.TestCase):

    def test_default_values(self):
        sr = StudentRisk(student_id=1)
        self.assertEqual(sr.student_id, 1)
        self.assertEqual(sr.risk_score, 0.0)
        self.assertEqual(sr.warning_level, "NORMAL")
        self.assertEqual(sr.last_activity, "Normal")
        self.assertEqual(sr.violations, 0)

    def test_last_update_is_recent(self):
        before = time.time()
        sr = StudentRisk(student_id=2)
        after = time.time()
        self.assertGreaterEqual(sr.last_update, before)
        self.assertLessEqual(sr.last_update, after)


class TestRiskEngineInit(unittest.TestCase):

    def setUp(self):
        self.engine = RiskEngine()

    def test_students_empty_on_init(self):
        self.assertEqual(len(self.engine.students), 0)

    def test_risk_table_has_expected_keys(self):
        expected = {
            "LOOK_LEFT", "LOOK_RIGHT", "LOOK_UP", "LOOK_DOWN",
            "HAND_FACE", "HAND_RAISED", "LEANING", "STANDING",
            "MULTIPLE_PERSON", "CELL_PHONE", "BOOK", "LAPTOP",
            "TALKING", "NORMAL"
        }
        self.assertEqual(set(self.engine.risk_table.keys()), expected)

    def test_normal_activity_decreases_risk(self):
        self.assertLess(self.engine.risk_table["NORMAL"], 0)

    def test_critical_activities_have_highest_scores(self):
        self.assertGreaterEqual(self.engine.risk_table["CELL_PHONE"], 50)
        self.assertGreaterEqual(self.engine.risk_table["MULTIPLE_PERSON"], 30)


class TestCreateStudent(unittest.TestCase):

    def setUp(self):
        self.engine = RiskEngine()

    def test_create_new_student(self):
        self.engine.create_student(10)
        self.assertIn(10, self.engine.students)

    def test_create_student_is_idempotent(self):
        self.engine.create_student(10)
        self.engine.update(10, "LOOK_LEFT")
        score_before = self.engine.students[10].risk_score
        self.engine.create_student(10)   # should not overwrite
        score_after = self.engine.students[10].risk_score
        self.assertEqual(score_before, score_after)


class TestRiskEngineUpdate(unittest.TestCase):

    def setUp(self):
        self.engine = RiskEngine()

    def test_update_increases_score_on_violation(self):
        self.engine.update(1, "LOOK_LEFT")
        self.assertGreater(self.engine.students[1].risk_score, 0)

    def test_update_increments_violations(self):
        self.engine.update(1, "CELL_PHONE")
        self.assertEqual(self.engine.students[1].violations, 1)

    def test_normal_activity_does_not_increment_violations(self):
        self.engine.update(1, "NORMAL")
        self.assertEqual(self.engine.students[1].violations, 0)

    def test_score_capped_at_max(self):
        for _ in range(10):
            self.engine.update(1, "MULTIPLE_PERSON")
        self.assertLessEqual(self.engine.students[1].risk_score, RiskEngine.MAX_RISK)

    def test_score_not_below_min(self):
        self.engine.update(1, "NORMAL")
        self.assertGreaterEqual(self.engine.students[1].risk_score, RiskEngine.MIN_RISK)

    def test_last_activity_updated(self):
        self.engine.update(1, "STANDING")
        self.assertEqual(self.engine.students[1].last_activity, "STANDING")

    def test_unknown_activity_adds_zero_points(self):
        self.engine.update(1, "UNKNOWN_ACTIVITY")
        self.assertEqual(self.engine.students[1].risk_score, 0.0)

    def test_case_insensitive_activity(self):
        self.engine.update(1, "look_left")
        self.assertGreater(self.engine.students[1].risk_score, 0)

    def test_auto_creates_student_on_update(self):
        self.assertNotIn(99, self.engine.students)
        self.engine.update(99, "LOOK_RIGHT")
        self.assertIn(99, self.engine.students)


class TestCalculateLevel(unittest.TestCase):

    def setUp(self):
        self.engine = RiskEngine()

    def test_normal_level(self):
        self.assertEqual(self.engine.calculate_level(0), "NORMAL")
        self.assertEqual(self.engine.calculate_level(19), "NORMAL")

    def test_low_level(self):
        self.assertEqual(self.engine.calculate_level(20), "LOW")
        self.assertEqual(self.engine.calculate_level(39), "LOW")

    def test_medium_level(self):
        self.assertEqual(self.engine.calculate_level(40), "MEDIUM")
        self.assertEqual(self.engine.calculate_level(59), "MEDIUM")

    def test_high_level(self):
        self.assertEqual(self.engine.calculate_level(60), "HIGH")
        self.assertEqual(self.engine.calculate_level(79), "HIGH")

    def test_critical_level(self):
        self.assertEqual(self.engine.calculate_level(80), "CRITICAL")
        self.assertEqual(self.engine.calculate_level(100), "CRITICAL")


class TestGetStudent(unittest.TestCase):

    def setUp(self):
        self.engine = RiskEngine()

    def test_get_existing_student(self):
        self.engine.create_student(5)
        student = self.engine.get_student(5)
        self.assertIsNotNone(student)
        self.assertEqual(student.student_id, 5)

    def test_get_nonexistent_student_returns_none(self):
        result = self.engine.get_student(999)
        self.assertIsNone(result)

    def test_get_all_students(self):
        self.engine.create_student(1)
        self.engine.create_student(2)
        all_students = self.engine.get_all_students()
        self.assertIn(1, all_students)
        self.assertIn(2, all_students)


class TestReset(unittest.TestCase):

    def setUp(self):
        self.engine = RiskEngine()

    def test_reset_zeroes_score_and_violations(self):
        self.engine.update(1, "CELL_PHONE")
        self.engine.update(1, "STANDING")
        self.engine.reset(1)
        student = self.engine.get_student(1)
        self.assertEqual(student.risk_score, 0)
        self.assertEqual(student.violations, 0)
        self.assertEqual(student.warning_level, "NORMAL")

    def test_reset_nonexistent_student_no_error(self):
        try:
            self.engine.reset(999)
        except Exception as e:
            self.fail(f"reset() raised an exception for missing student: {e}")


class TestDecay(unittest.TestCase):

    def setUp(self):
        self.engine = RiskEngine()

    def test_decay_reduces_score_after_timeout(self):
        self.engine.update(1, "CELL_PHONE")
        score_before = self.engine.students[1].risk_score
        self.engine.students[1].last_update -= 10
        self.engine.decay()
        score_after = self.engine.students[1].risk_score
        self.assertLess(score_after, score_before)

    def test_decay_does_not_go_below_zero(self):
        self.engine.create_student(1)
        self.engine.students[1].last_update -= 10
        self.engine.decay()
        self.assertGreaterEqual(self.engine.students[1].risk_score, 0)


if __name__ == "__main__":
    unittest.main()
