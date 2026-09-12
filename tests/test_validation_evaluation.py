import unittest

from validation.evaluation import (
    ExpectedEvent,
    calculate_metrics,
    evaluate_events,
    load_annotations,
    summarize_risk,
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


if __name__ == "__main__":
    unittest.main()
