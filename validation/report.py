"""HTML reporting for validation runs."""

import html
import json
from pathlib import Path
from typing import Any


def generate_validation_report(
    result: dict[str, Any],
    evaluation: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Write a readable local HTML report without claiming unmeasured accuracy."""
    evaluation = evaluation or {}
    overall = evaluation.get("overall", {})
    metrics = evaluation.get("metrics", {})
    false_positive_rows = "".join(
        f"<tr><td>{item.get('timestamp', 'n/a')}</td><td>{html.escape(str(item.get('student_id', 'n/a')))}</td>"
        f"<td>{html.escape(str(item.get('event', 'n/a')))}</td><td>{item.get('risk_score', 'n/a')}</td>"
        f"<td>{item.get('severity', 'n/a')}</td><td>{item.get('confidence', 'n/a')}</td></tr>"
        for item in evaluation.get("false_positives", [])
    ) or '<tr><td colspan="6">None measured.</td></tr>'
    false_negative_rows = "".join(
        f"<tr><td>{html.escape(str(item.get('scenario_id', 'n/a')))}</td><td>{html.escape(str(item.get('event', 'n/a')))}</td>"
        f"<td>{item.get('expected_window', [])}</td><td>{len(item.get('related_detections', []))}</td></tr>"
        for item in evaluation.get("false_negatives", [])
    ) or '<tr><td colspan="4">None measured.</td></tr>'
    rows = "".join(
        f"<tr><td>{html.escape(str(event))}</td><td>{data.get('true_positives', 0)}</td>"
        f"<td>{data.get('false_positives', 0)}</td><td>{data.get('false_negatives', 0)}</td>"
        f"<td>{data.get('precision', 'not applicable')}</td><td>{data.get('recall', 'not applicable')}</td>"
        f"<td>{data.get('f1', 'not applicable')}</td></tr>"
        for event, data in sorted(metrics.items())
    )
    if not rows:
        rows = '<tr><td colspan="7">No ground-truth event metrics available.</td></tr>'
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Validation Report</title>
<style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#17202a}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #ccd;padding:.5rem;text-align:left}}th{{background:#eef2f5}}code{{white-space:pre-wrap}}</style>
</head><body>
<h1>Detection Validation Report</h1>
<p><strong>Session:</strong> {html.escape(str(result.get('session', 'unknown')))}</p>
<p><strong>Video:</strong> {html.escape(str(result.get('video_file', 'unknown')))}</p>
<p><strong>Frames:</strong> {result.get('frames_processed', 0)} &nbsp; <strong>Duration:</strong> {result.get('duration_seconds', 0)}s &nbsp; <strong>Average FPS:</strong> {result.get('average_fps', 0)}</p>
<h2>Metrics</h2><p>Temporal tolerance: {evaluation.get('tolerance_seconds', 'not measured')} seconds.</p>
<p><strong>{'Measured against annotations.' if evaluation else 'No real-world accuracy claim: no annotated recording was evaluated.'}</strong></p>
<table><tr><th>Event</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1</th></tr>{rows}</table>
<p><strong>Overall:</strong> {html.escape(json.dumps(overall, sort_keys=True))}</p>
<h2>False Positives</h2><table><tr><th>Timestamp</th><th>Student</th><th>Event</th><th>Risk</th><th>Severity</th><th>Confidence</th></tr>{false_positive_rows}</table>
<h2>False Negatives</h2><table><tr><th>Scenario</th><th>Expected Event</th><th>Window</th><th>Related Detections</th></tr>{false_negative_rows}</table>
<h2>Scenario Summary</h2><pre><code>{html.escape(json.dumps(result.get('scenario_summaries', []), indent=2))}</code></pre>
<h2>Risk and Stability</h2><pre><code>{html.escape(json.dumps({'risk': result.get('risk'), 'stability': result.get('stability')}, indent=2))}</code></pre>
<h2>Configuration</h2><pre><code>{html.escape(json.dumps(result.get('configuration', {}), indent=2))}</code></pre>
<h2>Limitations</h2><p>Metrics are meaningful only when annotations correspond to this recording. No real-world accuracy claim is made without annotated video ground truth.</p>
</body></html>"""
    destination = Path(output_path or Path(result.get("result_file", "validation/results/result.json")).with_suffix(".html"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    return destination


def generate_tuning_report(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write an HTML comparison table for controlled threshold experiments."""
    table_rows = "".join(
        f"<tr><td>{html.escape(str(row.get('name')))}</td>"
        f"<td>{html.escape(json.dumps(row.get('parameter_changes', {}), sort_keys=True))}</td>"
        f"<td>{row.get('precision', 'insufficient data')}</td><td>{row.get('recall', 'insufficient data')}</td>"
        f"<td>{row.get('f1', 'insufficient data')}</td><td>{row.get('false_positives')}</td>"
        f"<td>{row.get('false_negatives')}</td></tr>"
        for row in rows
    ) or '<tr><td colspan="7">No tuning runs measured.</td></tr>'
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "<html><head><meta charset='utf-8'><title>Validation Tuning</title></head><body>"
        "<h1>Validation Tuning Comparison</h1>"
        "<p>These are experiment results, not automatic production-default changes.</p>"
        "<table border='1'><tr><th>Run</th><th>Parameter Changes</th><th>Precision</th>"
        "<th>Recall</th><th>F1</th><th>FP</th><th>FN</th></tr>"
        f"{table_rows}</table></body></html>",
        encoding="utf-8",
    )
    return destination
