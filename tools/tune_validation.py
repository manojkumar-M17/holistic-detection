#!/usr/bin/env python3
"""Run a small controlled validation parameter grid."""

import argparse
import json
from pathlib import Path

import config.config as cfg
from validation.evaluation import evaluate_events, load_annotations
from validation.runner import ValidationRunner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", default="validation/results/tuning")
    parser.add_argument("--yaw-thresholds", nargs="+", type=float, default=[cfg.HEAD_YAW_THRESHOLD])
    parser.add_argument("--tolerance", type=float, default=2.0)
    args = parser.parse_args()

    annotations = load_annotations(json.loads(Path(args.annotations).read_text(encoding="utf-8")))
    original = cfg.HEAD_YAW_THRESHOLD
    output = Path(args.output)
    summary = []
    try:
        for index, threshold in enumerate(args.yaw_thresholds, start=1):
            cfg.HEAD_YAW_THRESHOLD = threshold
            run_dir = output / f"run_{index:03d}"
            result = ValidationRunner(args.video, run_dir).run(session_name=f"run_{index:03d}")
            metrics = evaluate_events(annotations, result["detections"], args.tolerance)
            result["evaluation"] = metrics
            Path(result["result_file"]).write_text(json.dumps(result, indent=2), encoding="utf-8")
            summary.append({
                "run": f"run_{index:03d}",
                "yaw_threshold": threshold,
                "f1": metrics["overall"]["f1"],
                "false_positives": metrics["overall"]["false_positives"],
                "false_negatives": metrics["overall"]["false_negatives"],
            })
    finally:
        cfg.HEAD_YAW_THRESHOLD = original

    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Tuning summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
