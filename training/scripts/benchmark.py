#!/usr/bin/env python3
"""
Model Inference Latency & Throughput Benchmarker
-----------------------------------------------
Measures inference latency (mean, median, p95), frame throughput (FPS),
and resource consumption across CPU or GPU targets.
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def benchmark_model(
    model_path: str,
    imgsz: int = 640,
    iterations: int = 50,
    warmup: int = 10,
    device: str = "cpu",
) -> Dict[str, Any]:
    from ultralytics import YOLO

    print(f"Loading model '{model_path}' on device '{device}'...")
    model = YOLO(model_path)

    # Create dummy frame: (imgsz, imgsz, 3) uint8
    dummy_frame = np.random.randint(0, 256, (imgsz, imgsz, 3), dtype=np.uint8)

    print(f"Running {warmup} warmup iterations...")
    for _ in range(warmup):
        _ = model.predict(dummy_frame, imgsz=imgsz, device=device, verbose=False)

    print(f"Running {iterations} benchmark iterations...")
    latencies: List[float] = []

    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = model.predict(dummy_frame, imgsz=imgsz, device=device, verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # Convert to ms

    latencies_arr = np.array(latencies)
    mean_ms = float(np.mean(latencies_arr))
    median_ms = float(np.median(latencies_arr))
    p95_ms = float(np.percentile(latencies_arr, 95))
    min_ms = float(np.min(latencies_arr))
    max_ms = float(np.max(latencies_arr))
    std_ms = float(np.std(latencies_arr))
    fps = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0

    sys_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
    }

    try:
        import torch
        sys_info["torch_version"] = torch.__version__
        sys_info["cuda_available"] = torch.cuda.is_available()
    except Exception:
        pass

    results = {
        "model_path": str(model_path),
        "imgsz": imgsz,
        "device": device,
        "iterations": iterations,
        "warmup": warmup,
        "benchmarked_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "mean_latency_ms": round(mean_ms, 2),
            "median_latency_ms": round(median_ms, 2),
            "p95_latency_ms": round(p95_ms, 2),
            "min_latency_ms": round(min_ms, 2),
            "max_latency_ms": round(max_ms, 2),
            "std_latency_ms": round(std_ms, 2),
            "average_fps": round(fps, 1),
        },
        "system_info": sys_info,
    }

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark YOLO model inference latency and FPS.")
    parser.add_argument("--model", type=str, default="weights/yolov8n.pt", help="Path to model weights (.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image dimension")
    parser.add_argument("--iterations", type=int, default=50, help="Number of timed iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu', '0', etc.)")
    parser.add_argument("--output", type=str, default="training/reports/benchmark_results.json", help="Path to JSON output")

    args = parser.parse_args()

    model_p = Path(args.model)
    if not model_p.exists():
        print(f"Error: Model not found at {model_p}", file=sys.stderr)
        return 1

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    results = benchmark_model(
        model_path=str(model_p),
        imgsz=args.imgsz,
        iterations=args.iterations,
        warmup=args.warmup,
        device=args.device,
    )

    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    m = results["metrics"]
    print("\nBenchmark Results Summary:")
    print("=" * 45)
    print(f"  Model:            {results['model_path']}")
    print(f"  Device:           {results['device']}")
    print(f"  Image Size:       {results['imgsz']}x{results['imgsz']}")
    print(f"  Mean Latency:     {m['mean_latency_ms']} ms")
    print(f"  Median (p50):     {m['median_latency_ms']} ms")
    print(f"  95th %ile (p95):  {m['p95_latency_ms']} ms")
    print(f"  Min / Max:        {m['min_latency_ms']} / {m['max_latency_ms']} ms")
    print(f"  Average FPS:      {m['average_fps']} FPS")
    print("=" * 45)
    print(f"Report written to: {out_p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
