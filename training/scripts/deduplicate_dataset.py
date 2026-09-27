#!/usr/bin/env python3
"""
Dataset Deduplication & Near-Duplicate Detector
-----------------------------------------------
Scans image datasets for exact duplicates (SHA-256) and perceptual
near-duplicates using difference hash (dHash) with Hamming distance.

Prevents training/validation data leakage and redundant model training.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(img: np.ndarray, hash_size: int = 8) -> int:
    """Compute 64-bit difference hash (dHash) using OpenCV.
    
    1. Grayscale
    2. Resize to (hash_size + 1, hash_size), i.e. 9x8
    3. Compare adjacent pixels: row[x] > row[x+1]
    4. Convert 64 boolean bits into 64-bit unsigned integer
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Resize to (width=9, height=8)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    
    # Compute horizontal difference: col[:, 1:] > col[:, :-1]
    diff = resized[:, 1:] > resized[:, :-1]
    
    # Flatten diff array of bools and pack into integer
    hash_int = 0
    for bit in diff.flatten():
        hash_int = (hash_int << 1) | int(bit)
    return hash_int


def hamming_distance(h1: int, h2: int) -> int:
    """Count differing bits between two 64-bit integers."""
    return bin(h1 ^ h2).count("1")


def scan_dataset_duplicates(
    dataset_dir: Path,
    dhash_threshold: int = 4,
) -> Dict[str, Any]:
    images_dir = dataset_dir / "images" if (dataset_dir / "images").is_dir() else dataset_dir
    labels_dir = dataset_dir / "labels" if (dataset_dir / "labels").is_dir() else dataset_dir

    image_files = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])
    total_images = len(image_files)

    exact_hash_map: Dict[str, List[str]] = {}
    dhashes: List[Tuple[Path, int]] = []
    unreadable_images: List[str] = []

    for img_path in image_files:
        try:
            sha = compute_sha256(img_path)
            exact_hash_map.setdefault(sha, []).append(str(img_path))
        except Exception as e:
            unreadable_images.append(str(img_path))
            continue

        cv_img = cv2.imread(str(img_path))
        if cv_img is None:
            unreadable_images.append(str(img_path))
            continue

        dh = compute_dhash(cv_img)
        dhashes.append((img_path, dh))

    # Identify exact duplicate clusters
    exact_duplicates: List[List[str]] = [paths for paths in exact_hash_map.values() if len(paths) > 1]
    exact_duplicate_files_to_remove: Set[str] = set()
    for cluster in exact_duplicates:
        # Keep the first, flag rest as duplicate
        for p in cluster[1:]:
            exact_duplicate_files_to_remove.add(p)

    # Identify near-duplicate pairs (excluding already exact duplicates)
    near_duplicate_pairs: List[Dict[str, Any]] = []
    n_dhashes = len(dhashes)
    for i in range(n_dhashes):
        path_a, hash_a = dhashes[i]
        if str(path_a) in exact_duplicate_files_to_remove:
            continue
        for j in range(i + 1, n_dhashes):
            path_b, hash_b = dhashes[j]
            if str(path_b) in exact_duplicate_files_to_remove:
                continue
            dist = hamming_distance(hash_a, hash_b)
            if dist <= dhash_threshold:
                near_duplicate_pairs.append({
                    "image_a": str(path_a),
                    "image_b": str(path_b),
                    "hamming_distance": dist,
                })

    return {
        "dataset_dir": str(dataset_dir),
        "total_images_scanned": total_images,
        "unreadable_images": unreadable_images,
        "exact_duplicate_clusters_count": len(exact_duplicates),
        "exact_duplicate_clusters": exact_duplicates,
        "exact_duplicate_redundant_count": len(exact_duplicate_files_to_remove),
        "near_duplicate_pairs_count": len(near_duplicate_pairs),
        "near_duplicate_pairs": near_duplicate_pairs,
        "dhash_threshold": dhash_threshold,
    }


def execute_action(
    results: Dict[str, Any],
    action: str,
    quarantine_dir: Optional[Path] = None,
    dataset_dir: Optional[Path] = None,
) -> Dict[str, int]:
    if action == "report":
        return {"images_affected": 0, "labels_affected": 0}

    # Files to remove/move: all redundant exact duplicates + image_b in near duplicate pairs
    redundant_paths: Set[str] = set()
    for cluster in results.get("exact_duplicate_clusters", []):
        for p in cluster[1:]:
            redundant_paths.add(p)

    for pair in results.get("near_duplicate_pairs", []):
        redundant_paths.add(pair["image_b"])

    img_count = 0
    lbl_count = 0

    if action == "quarantine" and quarantine_dir:
        (quarantine_dir / "images").mkdir(parents=True, exist_ok=True)
        (quarantine_dir / "labels").mkdir(parents=True, exist_ok=True)

    for img_str in redundant_paths:
        img_p = Path(img_str)
        if not img_p.exists():
            continue

        # Find corresponding label
        lbl_p = None
        # Try parallel labels dir or same dir
        candidate1 = img_p.parent.parent / "labels" / f"{img_p.stem}.txt"
        candidate2 = img_p.with_suffix(".txt")
        if candidate1.is_file():
            lbl_p = candidate1
        elif candidate2.is_file():
            lbl_p = candidate2

        if action == "remove":
            img_p.unlink()
            img_count += 1
            if lbl_p and lbl_p.exists():
                lbl_p.unlink()
                lbl_count += 1
        elif action == "quarantine" and quarantine_dir:
            shutil.move(str(img_p), str(quarantine_dir / "images" / img_p.name))
            img_count += 1
            if lbl_p and lbl_p.exists():
                shutil.move(str(lbl_p), str(quarantine_dir / "labels" / lbl_p.name))
                lbl_count += 1

    return {"images_affected": img_count, "labels_affected": lbl_count}


def main() -> int:
    parser = argparse.ArgumentParser(description="Find and manage duplicate and near-duplicate images in dataset.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Directory containing images and labels")
    parser.add_argument("--dhash-threshold", type=int, default=4, help="Hamming distance threshold for near-duplicates (0-64, default 4)")
    parser.add_argument("--action", choices=["report", "quarantine", "remove"], default="report", help="Action to perform on duplicate items")
    parser.add_argument("--quarantine-dir", type=str, default=None, help="Directory to move duplicates to if action=quarantine")
    parser.add_argument("--output-report", type=str, default=None, help="Path to save JSON duplicate analysis report")

    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        print(f"Error: dataset path does not exist: {dataset_path}", file=sys.stderr)
        return 1

    if args.action == "quarantine" and not args.quarantine_dir:
        print("Error: --quarantine-dir must be specified when --action=quarantine", file=sys.stderr)
        return 1

    quarantine_path = Path(args.quarantine_dir) if args.quarantine_dir else None

    results = scan_dataset_duplicates(dataset_path, dhash_threshold=args.dhash_threshold)
    action_res = execute_action(results, action=args.action, quarantine_dir=quarantine_path, dataset_dir=dataset_path)

    results["action_taken"] = args.action
    results["action_stats"] = action_res

    report_path = Path(args.output_report) if args.output_report else dataset_path / "deduplication_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Deduplication Scan Complete:")
    print(f"  Scanned images: {results['total_images_scanned']}")
    print(f"  Exact duplicate clusters: {results['exact_duplicate_clusters_count']} (redundant files: {results['exact_duplicate_redundant_count']})")
    print(f"  Near-duplicate pairs (dHash <= {args.dhash_threshold}): {results['near_duplicate_pairs_count']}")
    print(f"  Action taken: {args.action} (images: {action_res['images_affected']}, labels: {action_res['labels_affected']})")
    print(f"  Report written to: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
