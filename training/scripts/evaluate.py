#!/usr/bin/env python3
"""
Model Evaluation & Baseline Comparison CLI
-----------------------------------------
Evaluates candidate custom YOLO models against the production baseline
on the canonical test split. Produces objective comparison metrics,
delta analyses, and JSON/HTML reports.

Enforces conservative deployment decision logic: the production baseline
is retained unless the candidate objectively outperforms it.
"""

import argparse
import html
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def evaluate_single_model(model_path: str, data_yaml: str, split: str = "test", device: str = "cpu") -> Dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, split=split, device=device, plots=False, verbose=False)

    summary: Dict[str, Any] = {
        "model_path": str(model_path),
        "split": split,
        "mAP50": float(getattr(metrics.box, "map50", 0.0)),
        "mAP50_95": float(getattr(metrics.box, "map", 0.0)),
        "precision": float(getattr(metrics.box, "mp", 0.0)),
        "recall": float(getattr(metrics.box, "mr", 0.0)),
        "per_class": {},
    }

    # Extract per-class mAP50 if available
    try:
        maps = getattr(metrics.box, "maps", None)
        names = metrics.names
        if maps is not None and names is not None:
            for idx, c_map in enumerate(maps):
                cls_name = names.get(idx, f"class_{idx}")
                summary["per_class"][cls_name] = {
                    "mAP50_95": float(c_map),
                }
    except Exception:
        pass

    return summary


def compare_models(candidate_metrics: Dict[str, Any], baseline_metrics: Dict[str, Any]) -> Dict[str, Any]:
    deltas = {
        "mAP50_delta": candidate_metrics["mAP50"] - baseline_metrics["mAP50"],
        "mAP50_95_delta": candidate_metrics["mAP50_95"] - baseline_metrics["mAP50_95"],
        "precision_delta": candidate_metrics["precision"] - baseline_metrics["precision"],
        "recall_delta": candidate_metrics["recall"] - baseline_metrics["recall"],
    }

    # Recommendation logic:
    # Candidate must improve mAP50 by at least +0.02 and not degrade precision significantly
    if deltas["mAP50_delta"] >= 0.02 and deltas["precision_delta"] >= -0.05:
        recommendation = "PROMOTE_CANDIDATE"
        reason = "Candidate model objectively outperforms baseline on the benchmark test split."
    elif deltas["mAP50_delta"] > -0.01:
        recommendation = "RETAIN_BASELINE"
        reason = "Candidate performance is comparable but does not demonstrate clear superiority over baseline."
    else:
        recommendation = "RETAIN_BASELINE"
        reason = "Candidate model underperforms baseline on the benchmark test split."

    return {
        "candidate": candidate_metrics,
        "baseline": baseline_metrics,
        "deltas": deltas,
        "recommendation": recommendation,
        "recommendation_rationale": reason,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "Benchmark evaluation on synthetic/curated test split. Does not constitute a real-world accuracy claim.",
    }


def generate_comparison_html(comparison_data: Dict[str, Any], output_path: Path) -> None:
    cand = comparison_data["candidate"]
    base = comparison_data["baseline"]
    deltas = comparison_data["deltas"]
    rec = comparison_data["recommendation"]
    reason = comparison_data["recommendation_rationale"]

    badge_color = "#28a745" if rec == "PROMOTE_CANDIDATE" else "#ffc107"
    badge_text_color = "#ffffff" if rec == "PROMOTE_CANDIDATE" else "#000000"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Model Comparison Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 30px; background: #f8f9fa; color: #212529; }}
        .container {{ max-width: 900px; margin: 0 auto; background: #fff; padding: 25px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ margin-top: 0; color: #343a40; }}
        .badge {{ display: inline-block; padding: 6px 12px; font-weight: bold; border-radius: 4px; background: {badge_color}; color: {badge_text_color}; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
        th {{ background: #e9ecef; }}
        .pos {{ color: #28a745; font-weight: bold; }}
        .neg {{ color: #dc3545; font-weight: bold; }}
        .disclaimer {{ margin-top: 25px; padding: 12px; background: #e8f4fd; border-left: 4px solid #007bff; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>YOLO Model Benchmark Comparison</h1>
        <p><strong>Evaluation Date:</strong> {html.escape(comparison_data['evaluated_at'])}</p>
        <p><strong>Recommendation:</strong> <span class="badge">{html.escape(rec)}</span></p>
        <p><em>{html.escape(reason)}</em></p>

        <h2>Summary Metrics</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Baseline Model ({html.escape(Path(base['model_path']).name)})</th>
                    <th>Candidate Model ({html.escape(Path(cand['model_path']).name)})</th>
                    <th>Delta</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>mAP@0.50</strong></td>
                    <td>{base['mAP50']:.4f}</td>
                    <td>{cand['mAP50']:.4f}</td>
                    <td class="{'pos' if deltas['mAP50_delta'] >= 0 else 'neg'}">{deltas['mAP50_delta']:+.4f}</td>
                </tr>
                <tr>
                    <td><strong>mAP@0.50:0.95</strong></td>
                    <td>{base['mAP50_95']:.4f}</td>
                    <td>{cand['mAP50_95']:.4f}</td>
                    <td class="{'pos' if deltas['mAP50_95_delta'] >= 0 else 'neg'}">{deltas['mAP50_95_delta']:+.4f}</td>
                </tr>
                <tr>
                    <td><strong>Precision</strong></td>
                    <td>{base['precision']:.4f}</td>
                    <td>{cand['precision']:.4f}</td>
                    <td class="{'pos' if deltas['precision_delta'] >= 0 else 'neg'}">{deltas['precision_delta']:+.4f}</td>
                </tr>
                <tr>
                    <td><strong>Recall</strong></td>
                    <td>{base['recall']:.4f}</td>
                    <td>{cand['recall']:.4f}</td>
                    <td class="{'pos' if deltas['recall_delta'] >= 0 else 'neg'}">{deltas['recall_delta']:+.4f}</td>
                </tr>
            </tbody>
        </table>

        <div class="disclaimer">
            <strong>Audit Note:</strong> {html.escape(comparison_data['disclaimer'])}
        </div>
    </div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate and compare candidate YOLO model against baseline.")
    parser.add_argument("--candidate-model", type=str, required=True, help="Path to candidate custom model weights (.pt)")
    parser.add_argument("--baseline-model", type=str, default="weights/yolov8n.pt", help="Path to baseline model weights (.pt)")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    parser.add_argument("--output-dir", type=str, default="training/reports/model_comparison", help="Output directory for reports")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device")

    args = parser.parse_args()

    cand_path = Path(args.candidate_model).resolve()
    base_path = Path(args.baseline_model).resolve()
    data_path = Path(args.data).resolve()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cand_path.is_file():
        print(f"Error: Candidate model not found at {cand_path}", file=sys.stderr)
        return 1

    if not base_path.is_file():
        print(f"Error: Baseline model not found at {base_path}", file=sys.stderr)
        return 1

    print(f"Evaluating candidate model: {cand_path}...")
    cand_metrics = evaluate_single_model(str(cand_path), str(data_path), split=args.split, device=args.device)

    print(f"Evaluating baseline model: {base_path}...")
    base_metrics = evaluate_single_model(str(base_path), str(data_path), split=args.split, device=args.device)

    comparison = compare_models(cand_metrics, base_metrics)

    json_path = out_dir / "model_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    html_path = out_dir / "model_comparison.html"
    generate_comparison_html(comparison, html_path)

    print(f"Comparison complete.")
    print(f"  Recommendation: {comparison['recommendation']}")
    print(f"  Rationale: {comparison['recommendation_rationale']}")
    print(f"  JSON report: {json_path}")
    print(f"  HTML report: {html_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
