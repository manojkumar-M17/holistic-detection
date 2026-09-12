#!/usr/bin/env python3
"""Run a small controlled validation parameter grid."""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path for direct CLI execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config.config as cfg
from validation.evaluation import compare_experiments, evaluate_events, load_annotation_file
from validation.report import generate_tuning_report
from validation.runner import ValidationInputError, ValidationRunner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", default="validation/results/tuning")
    parser.add_argument("--yaw-thresholds", nargs="+", type=float, default=[cfg.HEAD_YAW_THRESHOLD])
    parser.add_argument("--tolerance", type=float, default=2.0)
    args = parser.parse_args()

    try:
        annotations = load_annotation_file(args.annotations)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    original = cfg.HEAD_YAW_THRESHOLD
    output = Path(args.output)
    experiments = []
    try:
        for index, threshold in enumerate(args.yaw_thresholds, start=1):
            cfg.HEAD_YAW_THRESHOLD = threshold
            run_dir = output / f"run_{index:03d}"
            try:
                result = ValidationRunner(args.video, run_dir).run(session_name=f"run_{index:03d}")
            except ValidationInputError as exc:
                parser.error(str(exc))
                return 2
            metrics = evaluate_events(annotations, result["detections"], args.tolerance)
            result["evaluation"] = metrics
            Path(result["result_file"]).write_text(json.dumps(result, indent=2), encoding="utf-8")
            experiments.append({
                "name": f"run_{index:03d}",
                "parameter_changes": {"HEAD_YAW_THRESHOLD": threshold},
                "configuration": result["configuration"],
                "evaluation": metrics,
            })
    finally:
        cfg.HEAD_YAW_THRESHOLD = original

    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    comparison = compare_experiments(experiments)
    summary_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    generate_tuning_report(comparison, output / "summary.html")
    print(f"Tuning summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
