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
<table><tr><th>Event</th><th>TP</th><th>FP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1</th></tr>{rows}</table>
<p><strong>Overall:</strong> {html.escape(json.dumps(overall, sort_keys=True))}</p>
<h2>Risk and Stability</h2><pre><code>{html.escape(json.dumps({'risk': result.get('risk'), 'stability': result.get('stability')}, indent=2))}</code></pre>
<h2>Configuration</h2><pre><code>{html.escape(json.dumps(result.get('configuration', {}), indent=2))}</code></pre>
<h2>Limitations</h2><p>Metrics are meaningful only when annotations correspond to this recording. No real-world accuracy claim is made without annotated video ground truth.</p>
</body></html>"""
    destination = Path(output_path or Path(result.get("result_file", "validation/results/result.json")).with_suffix(".html"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    return destination
