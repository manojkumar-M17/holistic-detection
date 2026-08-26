"""
Tests for Rule Engine
---------------------
Tests for RuleEngine.evaluate() — event generation and risk engine integration.

Run:
    python -m pytest tests/test_rule_engine.py -v
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.risk_engine import RiskEngine
from modules.rule_engine import RuleEngine


class TestRuleEngineInit(unittest.TestCase):

    def test_rule_engine_stores_risk_engine(self):
        risk = RiskEngine()
        rule = RuleEngine(risk)
        self.assertIs(rule.risk_engine, risk)


class TestNormalBehaviour(unittest.TestCase):

    def setUp(self):
        self.risk = RiskEngine()
        self.rule = RuleEngine(self.risk)

    def test_empty_behaviour_returns_normal(self):
        events = self.rule.evaluate(1, {})
        self.assertIn("NORMAL", events)

    def test_head_center_returns_normal(self):
        events = self.rule.evaluate(1, {"head_direction": "CENTER"})
        self.assertIn("NORMAL", events)

    def test_all_false_flags_returns_normal(self):
        behaviour = {
            "head_direction": "CENTER",
            "standing": False,
            "phone": False,
            "multiple_person": False,
            "talking": False,
            "hand_face": False,
            "leaning": False,
        }
        events = self.rule.evaluate(1, behaviour)
        self.assertEqual(events, ["NORMAL"])


class TestHeadDirectionEvents(unittest.TestCase):

    def setUp(self):
        self.risk = RiskEngine()
        self.rule = RuleEngine(self.risk)

    def test_look_left_event(self):
        events = self.rule.evaluate(1, {"head_direction": "LEFT"})
        self.assertIn("LOOK_LEFT", events)

    def test_look_right_event(self):
        events = self.rule.evaluate(1, {"head_direction": "RIGHT"})
        self.assertIn("LOOK_RIGHT", events)

    def test_look_up_event(self):
        events = self.rule.evaluate(1, {"head_direction": "UP"})
        self.assertIn("LOOK_UP", events)

    def test_look_down_event(self):
        events = self.rule.evaluate(1, {"head_direction": "DOWN"})
        self.assertIn("LOOK_DOWN", events)

    def test_only_one_head_event_per_call(self):
        events = self.rule.evaluate(1, {"head_direction": "LEFT"})
        head_events = [e for e in events if e.startswith("LOOK_")]
        self.assertEqual(len(head_events), 1)


class TestSuspiciousFlags(unittest.TestCase):

    def setUp(self):
        self.risk = RiskEngine()
        self.rule = RuleEngine(self.risk)

    def test_standing_event(self):
        events = self.rule.evaluate(1, {"standing": True})
        self.assertIn("STANDING", events)

    def test_phone_event(self):
        events = self.rule.evaluate(1, {"phone": True})
        self.assertIn("CELL_PHONE", events)

    def test_multiple_person_event(self):
        events = self.rule.evaluate(1, {"multiple_person": True})
        self.assertIn("MULTIPLE_PERSON", events)

    def test_talking_event(self):
        events = self.rule.evaluate(1, {"talking": True})
        self.assertIn("TALKING", events)

    def test_hand_face_event(self):
        events = self.rule.evaluate(1, {"hand_face": True})
        self.assertIn("HAND_FACE", events)

    def test_leaning_event(self):
        events = self.rule.evaluate(1, {"leaning": True})
        self.assertIn("LEANING", events)


class TestMultipleEvents(unittest.TestCase):

    def setUp(self):
        self.risk = RiskEngine()
        self.rule = RuleEngine(self.risk)

    def test_multiple_flags_generate_multiple_events(self):
        behaviour = {
            "head_direction": "LEFT",
            "phone": True,
            "talking": True,
        }
        events = self.rule.evaluate(1, behaviour)
        self.assertIn("LOOK_LEFT", events)
        self.assertIn("CELL_PHONE", events)
        self.assertIn("TALKING", events)
        self.assertNotIn("NORMAL", events)

    def test_multiple_violations_update_risk_engine(self):
        behaviour = {"phone": True, "standing": True}
        self.rule.evaluate(1, behaviour)
        student = self.risk.get_student(1)
        self.assertGreater(student.risk_score, 0)
        self.assertGreater(student.violations, 0)


class TestRiskIntegration(unittest.TestCase):

    def setUp(self):
        self.risk = RiskEngine()
        self.rule = RuleEngine(self.risk)

    def test_evaluate_updates_risk_engine(self):
        self.rule.evaluate(1, {"head_direction": "LEFT"})
        student = self.risk.get_student(1)
        self.assertIsNotNone(student)
        self.assertGreater(student.risk_score, 0)

    def test_evaluate_creates_student_in_risk_engine(self):
        self.assertNotIn(42, self.risk.students)
        self.rule.evaluate(42, {"standing": True})
        self.assertIn(42, self.risk.students)

    def test_normal_behaviour_reduces_risk_over_time(self):
        self.rule.evaluate(1, {"head_direction": "LEFT"})
        score_after_violation = self.risk.get_student(1).risk_score
        self.rule.evaluate(1, {})  # NORMAL
        score_after_normal = self.risk.get_student(1).risk_score
        self.assertLess(score_after_normal, score_after_violation)


if __name__ == "__main__":
    unittest.main()
