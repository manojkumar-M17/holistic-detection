import unittest
import json
import tempfile
from pathlib import Path

from validation.evaluation import (
    ExpectedEvent,
    calculate_metrics,
    analyze_event_stability,
    compare_experiments,
    evaluate_events,
    load_annotations,
    load_annotation_file,
    summarize_risk,
    summarize_scenarios,
    summarize_stability,
)


class TestValidationEvaluation(unittest.TestCase):

    def test_load_annotations_validates_and_normalizes(self):
        annotations = load_annotations({
            "annotations": [{
                "scenario_id": "LOOK_LEFT",
                "start_time": 10,
                "end_time": 12,
                "expected_events": ["look_left"],
            }]
        })
        self.assertEqual(annotations[0].expected_events, ("LOOK_LEFT",))

    def test_annotation_file_errors_are_clear(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            load_annotation_file("missing-annotations.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.json"
            path.write_text("{bad", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not valid JSON"):
                load_annotation_file(path)

    def test_compare_experiments_returns_tuning_columns(self):
        rows = compare_experiments([
            {
                "name": "baseline",
                "parameter_changes": {},
                "evaluation": {
                    "overall": {
                        "precision": 0.5,
                        "recall": 1.0,
                        "f1": 2 / 3,
                        "false_positives": 1,
                        "false_negatives": 0,
                    }
                },
            }
        ])
        self.assertEqual(rows[0]["name"], "baseline")
        self.assertEqual(rows[0]["false_positives"], 1)
        self.assertAlmostEqual(rows[0]["f1"], 2 / 3)

    def test_scenario_risk_summary(self):
        summaries = summarize_scenarios(
            [ExpectedEvent("PHONE", 1, 3, ("CELL_PHONE",))],
            [{"timestamp": 2, "event": "CELL_PHONE", "risk_score": 95, "severity": "CRITICAL"}],
            tolerance_seconds=0,
        )
        self.assertEqual(summaries[0]["maximum_risk"], 95)
        self.assertEqual(summaries[0]["peak_severity"], "CRITICAL")

    def test_temporal_matching_and_metrics(self):
        annotations = [ExpectedEvent("LOOK_LEFT", 10, 12, ("LOOK_LEFT",))]
        result = evaluate_events(
            annotations,
            [
                {"timestamp": 13.5, "event": "LOOK_LEFT"},
                {"timestamp": 30, "event": "CELL_PHONE"},
            ],
            tolerance_seconds=2,
        )
        self.assertEqual(result["overall"]["true_positives"], 1)
        self.assertEqual(result["overall"]["false_positives"], 1)
        self.assertEqual(result["overall"]["false_negatives"], 0)
        self.assertEqual(result["overall"]["precision"], 0.5)
        self.assertEqual(result["metrics"]["LOOK_LEFT"]["recall"], 1.0)

    def test_false_negative_and_zero_denominator(self):
        result = evaluate_events(
            [ExpectedEvent("PHONE", 1, 2, ("CELL_PHONE",))],
            [],
            tolerance_seconds=0,
        )
        self.assertEqual(result["overall"]["false_negatives"], 1)
        self.assertIsNone(result["overall"]["precision"])
        self.assertEqual(calculate_metrics(0, 0, 0)["status"], "not_applicable")

    def test_invalid_annotation_and_duplicate_detection(self):
        with self.assertRaises(ValueError):
            load_annotations([{
                "scenario_id": "BAD",
                "start_time": 1,
                "end_time": 2,
                "expected_events": "CELL_PHONE",
            }])
        result = evaluate_events(
            [ExpectedEvent("PHONE", 1, 3, ("CELL_PHONE",))],
            [
                {"timestamp": 1.5, "event": "CELL_PHONE"},
                {"timestamp": 2.0, "event": "CELL_PHONE"},
            ],
            tolerance_seconds=0,
        )
        self.assertEqual(result["overall"]["true_positives"], 1)
        self.assertEqual(result["overall"]["false_positives"], 1)

    def test_event_stability_reports_latency_coverage_and_interruptions(self):
        annotation = ExpectedEvent("LOOK", 10, 20, ("LOOK_LEFT",))
        result = analyze_event_stability(
            [annotation],
            [
                {"timestamp": 12, "event": "LOOK_LEFT"},
                {"timestamp": 13, "event": "LOOK_LEFT"},
                {"timestamp": 18, "event": "LOOK_LEFT"},
            ],
            tolerance_seconds=0,
        )
        self.assertEqual(result[0]["detection_latency_seconds"], 2)
        self.assertEqual(result[0]["interruptions"], 1)
        self.assertGreater(result[0]["coverage_ratio"], 0)

    def test_stability_and_risk_summaries(self):
        detections = [
            {"timestamp": 1, "student_id": 1, "event": "LOOK_LEFT", "risk_score": 30, "severity": "LOW"},
            {"timestamp": 2, "student_id": 1, "event": "LOOK_LEFT", "risk_score": 40, "severity": "MEDIUM"},
        ]
        stability = summarize_stability(detections)
        self.assertEqual(stability[0]["detection_count"], 2)
        self.assertEqual(stability[0]["duration_seconds"], 1)
        risk = summarize_risk(detections)
        self.assertEqual(risk["minimum"], 30)
        self.assertEqual(risk["maximum"], 40)
        self.assertEqual(risk["severity_distribution"]["LOW"], 1)
        self.assertEqual(risk["status"], "measured")


if __name__ == "__main__":
    unittest.main()
