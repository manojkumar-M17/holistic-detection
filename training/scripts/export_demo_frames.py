#!/usr/bin/env python3
"""
Demo Frame Exporter
-------------------
Exports simulated frames from the demo monitoring stream or synthetic
generator for inspection, demonstration, or dataset augmentation.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

# Add project root to sys.path so config and modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.camera import CameraManager
from training.scripts.generate_synthetic import SyntheticExamSceneGenerator


def export_demo_frames(
    output_dir: Path,
    count: int = 20,
    use_synthetic_generator: bool = True,
    seed: int = 42,
) -> int:
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    if use_synthetic_generator:
        gen = SyntheticExamSceneGenerator(seed=seed)
        for i in range(count):
            img, annotations = gen.generate_scene()
            img_name = f"demo_frame_{i:04d}.jpg"
            lbl_name = f"demo_frame_{i:04d}.txt"

            cv2.imwrite(str(images_dir / img_name), img)
            with open(labels_dir / lbl_name, "w", encoding="utf-8") as f:
                for cls_id, xc, yc, w, h in annotations:
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    else:
        cam = CameraManager(source="demo")
        for i in range(count):
            ret, frame = cam.read_frame()
            if not ret or frame is None:
                break
            img_name = f"demo_frame_{i:04d}.jpg"
            lbl_name = f"demo_frame_{i:04d}.txt"
            cv2.imwrite(str(images_dir / img_name), frame)
            with open(labels_dir / lbl_name, "w", encoding="utf-8") as f:
                pass  # Background frame
        cam.release()

    meta = {
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_exported": count,
        "source": "synthetic_generator" if use_synthetic_generator else "demo_camera",
        "output_dir": str(output_dir),
    }
    with open(output_dir / "export_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Export demo / synthetic frames for evaluation or visualization.")
    parser.add_argument("--output-dir", type=str, default="training/datasets/demo_export", help="Output directory")
    parser.add_argument("--count", type=int, default=20, help="Number of frames to export")
    parser.add_argument("--seed", type=int, default=42, help="Seed for synthetic generation")
    parser.add_argument("--demo-cam", action="store_true", help="Use CameraManager demo stream instead of synthetic generator")

    args = parser.parse_args()
    out_path = Path(args.output_dir)
    print(f"Exporting {args.count} demo frames into {out_path}...")
    num = export_demo_frames(
        output_dir=out_path,
        count=args.count,
        use_synthetic_generator=not args.demo_cam,
        seed=args.seed,
    )
    print(f"Exported {num} frames successfully to {out_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
