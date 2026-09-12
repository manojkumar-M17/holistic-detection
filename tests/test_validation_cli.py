"""Tests for Validation and Tuning CLI tools."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import config.config as cfg
import tools.run_validation as run_validation_cli
import tools.tune_validation as tune_validation_cli


def _create_synthetic_video(path: str, num_frames: int = 2) -> None:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (640, 480))
    for _ in range(num_frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestValidationCLI(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.video_path = os.path.join(self.temp_dir, "synthetic.avi")
        _create_synthetic_video(self.video_path, num_frames=2)
        self.output_dir = os.path.join(self.temp_dir, "results")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_missing_video_exits_with_error(self):
        with patch.object(sys, "argv", ["run_validation.py", "--video", "nonexistent_file.avi"]):
            with self.assertRaises(SystemExit) as ctx:
                run_validation_cli.main()
            self.assertEqual(ctx.exception.code, 2)

    def test_invalid_video_exits_with_error(self):
        corrupt_video = os.path.join(self.temp_dir, "corrupted.avi")
        with open(corrupt_video, "w") as f:
            f.write("corrupt non-video data")
        with patch.object(sys, "argv", ["run_validation.py", "--video", corrupt_video]):
            with self.assertRaises(SystemExit) as ctx:
                run_validation_cli.main()
            self.assertEqual(ctx.exception.code, 2)

    def test_malformed_annotation_exits_with_error(self):
        bad_ann = os.path.join(self.temp_dir, "bad.json")
        with open(bad_ann, "w") as f:
            f.write("{invalid json")
        with patch.object(
            sys, "argv",
            ["run_validation.py", "--video", self.video_path, "--annotations", bad_ann, "--output", self.output_dir]
        ):
            with self.assertRaises(SystemExit) as ctx:
                run_validation_cli.main()
            self.assertEqual(ctx.exception.code, 2)

    def test_empty_annotation_list_succeeds(self):
        empty_ann = os.path.join(self.temp_dir, "empty.json")
        with open(empty_ann, "w") as f:
            json.dump([], f)
        with patch.object(
            sys, "argv",
            ["run_validation.py", "--video", self.video_path, "--annotations", empty_ann, "--output", self.output_dir]
        ):
            exit_code = run_validation_cli.main()
            self.assertEqual(exit_code, 0)

        json_files = list(Path(self.output_dir).glob("*.json"))
        html_files = list(Path(self.output_dir).glob("*.html"))
        self.assertEqual(len(json_files), 1)
        self.assertEqual(len(html_files), 1)

    def test_two_frame_synthetic_video_with_multiple_expected_events(self):
        ann_path = os.path.join(self.temp_dir, "multiple_events.json")
        with open(ann_path, "w") as f:
            json.dump([
                {
                    "scenario_id": "multi_event_scenario",
                    "start_time": 0.0,
                    "end_time": 0.5,
                    "expected_events": ["LOOK_LEFT", "CELL_PHONE", "MULTIPLE_PERSON"],
                }
            ], f)
        with patch.object(
            sys, "argv",
            ["run_validation.py", "--video", self.video_path, "--annotations", ann_path, "--output", self.output_dir]
        ):
            exit_code = run_validation_cli.main()
            self.assertEqual(exit_code, 0)

        result_path = Path(self.output_dir) / "synthetic.json"
        self.assertTrue(result_path.is_file())
        data = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(data["frames_processed"], 2)
        self.assertIn("evaluation", data)
        self.assertIn("overall", data["evaluation"])
        # Expected events not detected in black frames become false negatives
        self.assertEqual(data["evaluation"]["overall"]["false_negatives"], 3)


class TestTuningCLI(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.video_path = os.path.join(self.temp_dir, "synthetic.avi")
        _create_synthetic_video(self.video_path, num_frames=2)
        self.output_dir = os.path.join(self.temp_dir, "tuning")
        self.ann_path = os.path.join(self.temp_dir, "ann.json")
        with open(self.ann_path, "w") as f:
            json.dump([
                {
                    "scenario_id": "tuning_scenario",
                    "start_time": 0.0,
                    "end_time": 0.5,
                    "expected_events": ["LOOK_LEFT"],
                }
            ], f)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_missing_input_exits_with_error(self):
        with patch.object(
            sys, "argv",
            ["tune_validation.py", "--video", "missing.avi", "--annotations", self.ann_path]
        ):
            with self.assertRaises(SystemExit) as ctx:
                tune_validation_cli.main()
            self.assertEqual(ctx.exception.code, 2)

    def test_missing_annotations_exits_with_error(self):
        with patch.object(
            sys, "argv",
            ["tune_validation.py", "--video", self.video_path, "--annotations", "missing.json"]
        ):
            with self.assertRaises(SystemExit) as ctx:
                tune_validation_cli.main()
            self.assertEqual(ctx.exception.code, 2)

    def test_tuning_runs_and_preserves_production_defaults(self):
        original_yaw = cfg.HEAD_YAW_THRESHOLD
        with patch.object(
            sys, "argv",
            [
                "tune_validation.py",
                "--video", self.video_path,
                "--annotations", self.ann_path,
                "--output", self.output_dir,
                "--yaw-thresholds", "20.0", "30.0",
            ]
        ):
            exit_code = tune_validation_cli.main()
            self.assertEqual(exit_code, 0)

        # Production defaults must not be changed
        self.assertEqual(cfg.HEAD_YAW_THRESHOLD, original_yaw)

        summary_json = Path(self.output_dir) / "summary.json"
        summary_html = Path(self.output_dir) / "summary.html"
        self.assertTrue(summary_json.is_file())
        self.assertTrue(summary_html.is_file())

        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        self.assertIsInstance(summary, list)
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0]["name"], "run_001")


if __name__ == "__main__":
    unittest.main()

