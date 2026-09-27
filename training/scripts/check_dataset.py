#!/usr/bin/env python3
"""
Dataset Quality Inspector
-------------------------
Validates YOLO dataset integrity:
- Checks image readability and dimensions.
- Validates bounding box coordinates (normalized in [0, 1], non-zero area).
- Detects orphan images (missing label) and orphan labels (missing image).
- Flags corrupt images or malformed label syntax.
- Produces machine-readable JSON and printable HTML quality reports.
"""

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def inspect_dataset(dataset_dir: str | os.PathLike[str], max_classes: int = 8) -> Dict[str, Any]:
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_dir}")

    # Discover images and labels
    # Supported structures:
    # 1. dataset/images/ and dataset/labels/
    # 2. flat dataset/ containing both .jpg and .txt
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"

    if images_dir.is_dir() and labels_dir.is_dir():
        image_files = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])
        label_files = sorted([p for p in labels_dir.rglob("*.txt")])
    else:
        image_files = sorted([p for p in dataset_path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and "reports" not in p.parts])
        label_files = sorted([p for p in dataset_path.rglob("*.txt") if p.name != "classes.txt" and "reports" not in p.parts])

    image_stems = {p.stem: p for p in image_files}
    label_stems = {p.stem: p for p in label_files}

    orphan_images = [str(p) for stem, p in image_stems.items() if stem not in label_stems]
    orphan_labels = [str(p) for stem, p in label_stems.items() if stem not in image_stems]

    valid_images = 0
    corrupt_images = []
    total_annotations = 0
    valid_annotations = 0
    invalid_annotations = []
    class_counts: Dict[int, int] = {}
    box_widths: List[float] = []
    box_heights: List[float] = []

    for stem, img_path in image_stems.items():
        # 1. Validate image readability
        try:
            img = cv2.imread(str(img_path))
            if img is None or img.size == 0:
                corrupt_images.append({"file": str(img_path), "error": "cv2.imread returned None or empty"})
                continue
            h, w = img.shape[:2]
            if h <= 0 or w <= 0:
                corrupt_images.append({"file": str(img_path), "error": f"Invalid dimensions ({w}x{h})"})
                continue
            valid_images += 1
        except Exception as e:
            corrupt_images.append({"file": str(img_path), "error": str(e)})
            continue

        # 2. Validate corresponding label file if present
        label_path = label_stems.get(stem)
        if not label_path or not label_path.exists():
            continue

        try:
            with open(label_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
        except Exception as e:
            invalid_annotations.append({"file": str(label_path), "error": f"Read error: {e}"})
            continue

        for line_num, line in enumerate(lines, start=1):
            total_annotations += 1
            parts = line.split()
            if len(parts) != 5:
                invalid_annotations.append({
                    "file": str(label_path),
                    "line": line_num,
                    "content": line,
                    "error": f"Expected 5 fields, got {len(parts)}"
                })
                continue

            try:
                cls_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
            except ValueError as e:
                invalid_annotations.append({
                    "file": str(label_path),
                    "line": line_num,
                    "content": line,
                    "error": f"Numeric parse error: {e}"
                })
                continue

            # Class ID range
            if cls_id < 0 or (max_classes is not None and cls_id >= max_classes):
                invalid_annotations.append({
                    "file": str(label_path),
                    "line": line_num,
                    "content": line,
                    "error": f"Class ID {cls_id} out of canonical range [0, {max_classes - 1}]"
                })
                continue

            # Normalized coordinate bounds check
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                invalid_annotations.append({
                    "file": str(label_path),
                    "line": line_num,
                    "content": line,
                    "error": f"Center coordinate ({xc}, {yc}) outside [0.0, 1.0]"
                })
                continue

            # Positive dimension check
            if bw <= 0.0 or bh <= 0.0:
                invalid_annotations.append({
                    "file": str(label_path),
                    "line": line_num,
                    "content": line,
                    "error": f"Zero or negative box dimension ({bw}x{bh})"
                })
                continue

            # Outer boundary check
            x1 = xc - bw / 2.0
            y1 = yc - bh / 2.0
            x2 = xc + bw / 2.0
            y2 = yc + bh / 2.0

            if x1 < -0.05 or y1 < -0.05 or x2 > 1.05 or y2 > 1.05:
                invalid_annotations.append({
                    "file": str(label_path),
                    "line": line_num,
                    "content": line,
                    "error": f"Box [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}] severely outside frame bounds"
                })
                continue

            valid_annotations += 1
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
            box_widths.append(bw)
            box_heights.append(bh)

    summary = {
        "dataset_path": str(dataset_path),
        "total_images_found": len(image_files),
        "valid_images": valid_images,
        "corrupt_images_count": len(corrupt_images),
        "total_label_files": len(label_files),
        "orphan_images_count": len(orphan_images),
        "orphan_labels_count": len(orphan_labels),
        "total_annotations": total_annotations,
        "valid_annotations": valid_annotations,
        "invalid_annotations_count": len(invalid_annotations),
        "class_distribution": class_counts,
        "box_size_stats": {
            "mean_width": round(float(np.mean(box_widths)), 4) if box_widths else 0.0,
            "mean_height": round(float(np.mean(box_heights)), 4) if box_heights else 0.0,
            "min_width": round(float(np.min(box_widths)), 4) if box_widths else 0.0,
            "max_width": round(float(np.max(box_widths)), 4) if box_widths else 0.0,
        },
        "issues": {
            "corrupt_images": corrupt_images[:50],
            "orphan_images_sample": orphan_images[:50],
            "orphan_labels_sample": orphan_labels[:50],
            "invalid_annotations_sample": invalid_annotations[:50],
        },
        "status": "PASS" if len(corrupt_images) == 0 and len(invalid_annotations) == 0 else "WARNINGS_FOUND"
    }

    return summary


def render_html_report(summary: Dict[str, Any], output_path: str | os.PathLike[str]) -> str:
    safe_dataset = html.escape(summary["dataset_path"])
    status = summary["status"]
    status_color = "#10b981" if status == "PASS" else "#f59e0b"

    class_rows = ""
    for cls_id, count in sorted(summary["class_distribution"].items()):
        class_rows += f"<tr><td>Class {cls_id}</td><td><strong>{count}</strong></td></tr>"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Dataset Quality Inspection Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 30px; }}
        .card {{ background: #1e293b; border-radius: 8px; padding: 24px; max-width: 900px; margin: 0 auto; box-shadow: 0 4px 20px rgba(0,0,0,0.4); }}
        h1 {{ margin-top: 0; color: #38bdf8; font-size: 24px; }}
        .badge {{ background: {status_color}; color: #ffffff; padding: 4px 12px; border-radius: 12px; font-weight: bold; font-size: 13px; }}
        .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 20px 0; }}
        .stat {{ background: #0f172a; padding: 12px; border-radius: 6px; text-align: center; border: 1px solid #334155; }}
        .stat-val {{ font-size: 20px; font-weight: bold; color: #38bdf8; }}
        .stat-lbl {{ font-size: 11px; text-transform: uppercase; color: #94a3b8; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ padding: 8px 12px; border-bottom: 1px solid #334155; text-align: left; font-size: 13px; }}
        th {{ background: #0f172a; color: #94a3b8; font-size: 11px; text-transform: uppercase; }}
    </style>
</head>
<body>
    <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <h1>Dataset Quality Inspection</h1>
            <span class="badge">{status}</span>
        </div>
        <p style="color:#94a3b8; font-size:13px;">Target: <code>{safe_dataset}</code></p>

        <div class="grid">
            <div class="stat"><div class="stat-lbl">Valid Images</div><div class="stat-val">{summary["valid_images"]}</div></div>
            <div class="stat"><div class="stat-lbl">Corrupt Images</div><div class="stat-val" style="color:#ef4444;">{summary["corrupt_images_count"]}</div></div>
            <div class="stat"><div class="stat-lbl">Valid Annotations</div><div class="stat-val">{summary["valid_annotations"]}</div></div>
            <div class="stat"><div class="stat-lbl">Invalid Annotations</div><div class="stat-val" style="color:#ef4444;">{summary["invalid_annotations_count"]}</div></div>
        </div>

        <h3 style="color:#38bdf8;">Class Distribution</h3>
        <table>
            <thead><tr><th>Class ID</th><th>Count</th></tr></thead>
            <tbody>{class_rows if class_rows else '<tr><td colspan="2">No annotations</td></tr>'}</tbody>
        </table>
    </div>
</body>
</html>
"""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(html_content, encoding="utf-8")
    return str(out_file)


def main():
    parser = argparse.ArgumentParser(description="Check YOLO dataset quality and label integrity")
    parser.add_argument("--dataset", "--dataset-dir", dest="dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--output-dir", default="training/reports", help="Directory for reports")
    parser.add_argument("--max-classes", type=int, default=8, help="Maximum allowed canonical classes")
    args = parser.parse_args()

    try:
        summary = inspect_dataset(args.dataset, max_classes=args.max_classes)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "dataset_quality.json"
        html_path = out_dir / "dataset_quality.html"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        render_html_report(summary, html_path)

        print(f"[CHECK] Summary: {summary['valid_images']} valid images, {summary['valid_annotations']} annotations.")
        print(f"[CHECK] Reports generated: {json_path} and {html_path}")
        if summary["corrupt_images_count"] > 0 or summary["invalid_annotations_count"] > 0:
            print(f"[CHECK] Warning: Issues detected ({summary['corrupt_images_count']} corrupt, {summary['invalid_annotations_count']} invalid bboxes)")

    except Exception as e:
        print(f"[CHECK] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
