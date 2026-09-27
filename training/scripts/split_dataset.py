#!/usr/bin/env python3
"""
Group-Aware Dataset Splitter
---------------------------
Splits a normalized YOLO dataset into train, val, and test sets:
- Enforces group/source/session isolation to prevent data leakage.
- Generates Ultralytics-compatible dataset.yaml and split_manifest.json.
- Checks class balance across splits.
"""

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_CLASS_NAMES = {
    0: "person",
    1: "phone",
    2: "paper",
    3: "book",
    4: "calculator",
    5: "laptop",
    6: "watch",
    7: "earphone_or_earbud",
}


def extract_group_id(stem: str) -> str:
    """Extract grouping key from image filename stem.
    
    Examples:
      - synth_000042 -> synth (or synth_000 for sub-groups)
      - session1_frame0123 -> session1
      - studentA_cam1_001 -> studentA
    """
    parts = stem.split("_")
    if len(parts) >= 2:
        # If it's synthetic with numbered sequence, group in chunks of 5
        if parts[0] == "synth" and parts[1].isdigit():
            idx = int(parts[1])
            return f"synth_chunk_{idx // 5}"
        return parts[0]
    return stem


def parse_label_file(label_path: Path) -> List[int]:
    """Extract list of class IDs present in a YOLO label file."""
    classes = []
    if not label_path.is_file():
        return classes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                try:
                    classes.append(int(parts[0]))
                except ValueError:
                    pass
    return classes


def split_dataset(
    input_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    copy_files: bool = True,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-4:
        raise ValueError(f"Split ratios must sum to 1.0 (got {train_ratio + val_ratio + test_ratio})")

    images_in = input_dir / "images" if (input_dir / "images").is_dir() else input_dir
    labels_in = input_dir / "labels" if (input_dir / "labels").is_dir() else input_dir

    image_files = sorted([p for p in images_in.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])
    if not image_files:
        raise ValueError(f"No images found in {images_in}")

    import random
    rng = random.Random(seed)

    # Group images
    groups: Dict[str, List[Path]] = {}
    for img_path in image_files:
        grp = extract_group_id(img_path.stem)
        groups.setdefault(grp, []).append(img_path)

    # Shuffle groups deterministically
    group_keys = sorted(list(groups.keys()))
    rng.shuffle(group_keys)

    total_images = len(image_files)
    target_train = int(round(total_images * train_ratio))
    target_val = int(round(total_images * val_ratio))
    
    train_images: List[Path] = []
    val_images: List[Path] = []
    test_images: List[Path] = []

    curr_train = 0
    curr_val = 0

    for grp in group_keys:
        grp_imgs = groups[grp]
        if curr_train + len(grp_imgs) <= target_train or (curr_train == 0 and len(train_images) == 0):
            train_images.extend(grp_imgs)
            curr_train += len(grp_imgs)
        elif curr_val + len(grp_imgs) <= target_val or (curr_val == 0 and len(val_images) == 0):
            val_images.extend(grp_imgs)
            curr_val += len(grp_imgs)
        else:
            test_images.extend(grp_imgs)

    # Ensure at least 1 in val and test if total images >= 3
    if total_images >= 3:
        if not val_images and len(train_images) > 1:
            val_images.append(train_images.pop())
        if not test_images and len(train_images) > 1:
            test_images.append(train_images.pop())

    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images,
    }

    # Setup directories
    for split_name in ["train", "val", "test"]:
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    manifest_entries: Dict[str, str] = {}
    split_stats: Dict[str, Dict[str, Any]] = {}

    for split_name, img_list in splits.items():
        class_distribution: Dict[int, int] = {}
        for img_path in img_list:
            manifest_entries[img_path.name] = split_name
            # Find label
            lbl_candidate = labels_in / f"{img_path.stem}.txt"
            if not lbl_candidate.is_file():
                matches = list(labels_in.rglob(f"{img_path.stem}.txt"))
                lbl_candidate = matches[0] if matches else None

            # Copy/symlink image
            dest_img = output_dir / "images" / split_name / img_path.name
            if copy_files:
                if not dest_img.exists() or dest_img.resolve() != img_path.resolve():
                    shutil.copy2(img_path, dest_img)
            else:
                if not dest_img.exists():
                    os.symlink(img_path.resolve(), dest_img)

            # Copy/symlink label
            dest_lbl = output_dir / "labels" / split_name / f"{img_path.stem}.txt"
            if lbl_candidate and lbl_candidate.is_file():
                lbl_classes = parse_label_file(lbl_candidate)
                for c in lbl_classes:
                    class_distribution[c] = class_distribution.get(c, 0) + 1

                if copy_files:
                    if not dest_lbl.exists() or dest_lbl.resolve() != lbl_candidate.resolve():
                        shutil.copy2(lbl_candidate, dest_lbl)
                else:
                    if not dest_lbl.exists():
                        os.symlink(lbl_candidate.resolve(), dest_lbl)
            else:
                # Empty label for background
                with open(dest_lbl, "w", encoding="utf-8") as f:
                    pass

        split_stats[split_name] = {
            "image_count": len(img_list),
            "percentage": round(len(img_list) / float(total_images) * 100, 2) if total_images > 0 else 0,
            "class_distribution": class_distribution,
        }

    # Generate dataset.yaml
    names_map = class_names or DEFAULT_CLASS_NAMES
    dataset_yaml_data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {int(k): v for k, v in names_map.items()},
    }

    yaml_file = output_dir / "dataset.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(dataset_yaml_data, f, sort_keys=False)

    manifest_data = {
        "seed": seed,
        "total_images": total_images,
        "requested_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "split_stats": split_stats,
        "file_split_map": manifest_entries,
    }

    manifest_file = output_dir / "split_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Group-aware dataset splitter for YOLO training.")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory containing images and labels")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory to place split dataset")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train ratio (default 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio (default 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio (default 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--symlink", action="store_true", help="Symlink files instead of copying")

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)

    print(f"Splitting dataset from {input_path} into {output_path}...")
    manifest = split_dataset(
        input_dir=input_path,
        output_dir=output_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        copy_files=not args.symlink,
    )

    print("Splitting complete:")
    for split_name, stats in manifest["split_stats"].items():
        print(f"  {split_name}: {stats['image_count']} images ({stats['percentage']}%) - classes: {stats['class_distribution']}")
    print(f"  dataset.yaml written to: {output_path / 'dataset.yaml'}")
    print(f"  split_manifest.json written to: {output_path / 'split_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
