#!/usr/bin/env python3
"""Run one local recording through the production detection pipeline."""

import argparse
import json
from pathlib import Path

from validation.evaluation import evaluate_events, load_annotations, summarize_risk, summarize_stability
from validation.report import generate_validation_report
from validation.runner import ValidationInputError, ValidationRunner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Local MP4, AVI, or MOV recording")
    parser.add_argument("--annotations", help="JSON annotation file")
    parser.add_argument("--output", default="validation/results", help="Output directory")
    parser.add_argument("--tolerance", type=float, default=2.0, help="Matching tolerance in seconds")
    args = parser.parse_args()

    try:
        result = ValidationRunner(args.video, args.output).run()
    except ValidationInputError as exc:
        parser.error(str(exc))
        return 2

    result["risk"] = summarize_risk(result["detections"])
    result["stability"] = summarize_stability(result["detections"])
    evaluation = None
    if args.annotations:
        annotation_data = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
        evaluation = evaluate_events(
            load_annotations(annotation_data),
            result["detections"],
            tolerance_seconds=args.tolerance,
        )
        result["evaluation"] = evaluation

    result_path = Path(result["result_file"])
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path = generate_validation_report(
        result,
        evaluation,
        result_path.with_suffix(".html"),
    )
    print(f"Results: {result_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
