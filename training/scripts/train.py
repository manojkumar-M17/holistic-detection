#!/usr/bin/env python3
"""
Custom YOLO Training CLI
------------------------
Trains custom YOLO models on the canonical exam dataset using Ultralytics.
Provides reproducible configuration, automatic device selection,
smoke test execution for CPU verification, and versioned artifact packaging.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def detect_best_device() -> str:
    """Detect available compute device (cuda > mps > cpu)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_training_config(config_path: Path) -> Dict[str, Any]:
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def train_yolo(
    data_yaml: Path,
    base_model: str = "weights/yolov8n.pt",
    epochs: int = 50,
    batch_size: int = 16,
    imgsz: int = 640,
    device: Optional[str] = None,
    output_name: str = "exam_yolov8n_v1",
    project_dir: Optional[Path] = None,
    smoke_test: bool = False,
    extra_hyperparams: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from ultralytics import YOLO

    # Resolve paths
    data_path = Path(data_yaml).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"dataset.yaml not found at: {data_path}")

    # Resolve base model
    model_str = base_model
    if not os.path.exists(model_str):
        # Fallback to weights/yolov8n.pt or yolov8n.pt
        alt = PROJECT_ROOT / "weights" / "yolov8n.pt"
        if alt.is_file():
            model_str = str(alt)
        else:
            model_str = "yolov8n.pt"

    project_path = project_dir or (PROJECT_ROOT / "training" / "runs" / "detect")
    project_path.mkdir(parents=True, exist_ok=True)

    target_device = device or detect_best_device()

    if smoke_test:
        print("[SMOKE TEST MODE ENABLED]")
        epochs = 1
        batch_size = 2
        imgsz = 320
        target_device = "cpu"

    print(f"Initializing YOLO model from '{model_str}'...")
    model = YOLO(model_str)

    train_kwargs: Dict[str, Any] = {
        "data": str(data_path),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": imgsz,
        "device": target_device,
        "project": str(project_path),
        "name": output_name,
        "exist_ok": True,
        "plots": not smoke_test,
        "verbose": True,
    }

    if smoke_test:
        train_kwargs["workers"] = 0
        train_kwargs["save"] = True

    if extra_hyperparams:
        for k, v in extra_hyperparams.items():
            if k not in train_kwargs and v is not None:
                train_kwargs[k] = v

    print(f"Beginning training with kwargs:")
    for k, v in train_kwargs.items():
        print(f"  {k}: {v}")

    start_time = datetime.now(timezone.utc)
    results = model.train(**train_kwargs)
    end_time = datetime.now(timezone.utc)

    # Package output model and metadata
    exp_dir = project_path / output_name
    weights_dir = exp_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"

    custom_model_dir = PROJECT_ROOT / "models" / "custom" / output_name
    custom_model_dir.mkdir(parents=True, exist_ok=True)

    if best_pt.is_file():
        shutil.copy2(best_pt, custom_model_dir / "best.pt")
    elif last_pt.is_file():
        shutil.copy2(last_pt, custom_model_dir / "best.pt")

    if last_pt.is_file():
        shutil.copy2(last_pt, custom_model_dir / "last.pt")

    # Extract final validation metrics if available
    metrics_summary: Dict[str, Any] = {}
    try:
        if hasattr(results, "results_dict"):
            metrics_summary = {str(k): float(v) for k, v in results.results_dict.items() if isinstance(v, (int, float))}
    except Exception:
        pass

    metadata = {
        "model_name": output_name,
        "base_model": str(model_str),
        "training_start": start_time.isoformat(),
        "training_end": end_time.isoformat(),
        "training_duration_seconds": (end_time - start_time).total_seconds(),
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "imgsz": imgsz,
        "device": target_device,
        "smoke_test": smoke_test,
        "git_commit": get_git_commit(),
        "dataset_yaml": str(data_path),
        "metrics_summary": metrics_summary,
        "status": "candidate_model",
    }

    meta_file = custom_model_dir / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Training completed successfully.")
    print(f"  Weights saved to: {custom_model_dir}")
    print(f"  Metadata saved to: {meta_file}")

    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Custom YOLO Training CLI.")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml")
    parser.add_argument("--model", type=str, default="weights/yolov8n.pt", help="Base model path or name")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', '0', etc.)")
    parser.add_argument("--name", type=str, default="exam_yolov8n_v1", help="Experiment name")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 1-epoch CPU smoke test")
    parser.add_argument("--config", type=str, default="training/config/training.yaml", help="Path to training config YAML")

    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_training_config(cfg_path)
    train_cfg = cfg.get("training", {})

    epochs = args.epochs or train_cfg.get("epochs", 50)
    batch = args.batch or train_cfg.get("batch_size", 16)
    imgsz = args.imgsz or train_cfg.get("imgsz", 640)
    device = args.device or train_cfg.get("device", None)

    try:
        train_yolo(
            data_yaml=Path(args.data),
            base_model=args.model,
            epochs=epochs,
            batch_size=batch,
            imgsz=imgsz,
            device=device,
            output_name=args.name,
            smoke_test=args.smoke_test,
        )
        return 0
    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
