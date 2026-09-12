import json
import tempfile
import unittest
from pathlib import Path

from validation.report import generate_tuning_report, generate_validation_report


class TestValidationReport(unittest.TestCase):

    def test_report_serializes_result_and_metrics(self):
        result = {
            "session": "synthetic",
            "video_file": "local.mp4",
            "frames_processed": 3,
            "duration_seconds": 1.0,
            "average_fps": 3.0,
            "configuration": {"HEAD_YAW_THRESHOLD": 25},
            "risk": {"minimum": 0, "maximum": 40},
            "stability": [],
        }
        evaluation = {
            "tolerance_seconds": 2.0,
            "overall": {"true_positives": 1, "false_positives": 0, "false_negatives": 0},
            "metrics": {"LOOK_LEFT": {"precision": 1.0, "recall": 1.0, "f1": 1.0}},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "report.html"
            generated = generate_validation_report(result, evaluation, destination)
            content = generated.read_text(encoding="utf-8")
            self.assertIn("synthetic", content)
            self.assertIn("LOOK_LEFT", content)
            self.assertTrue(generated.is_file())

    def test_tuning_report_serializes_comparison_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            generated = generate_tuning_report([
                {"name": "baseline", "parameter_changes": {}, "precision": 1.0,
                 "recall": 0.5, "f1": 2 / 3, "false_positives": 0, "false_negatives": 1}
            ], Path(temp_dir) / "tuning.html")
            self.assertIn("baseline", generated.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
