#!/usr/bin/env python3
"""
Dataset Label Normalizer
------------------------
Normalizes heterogeneous dataset annotations (COCO JSON or YOLO format)
into the project's canonical YOLO object taxonomy.

Canonical Object Classes:
  0: person
  1: phone
  2: paper
  3: book
  4: calculator
  5: laptop
  6: watch
  7: earphone_or_earbud
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_label_mapping(mapping_path: str | Path) -> Dict[str, Any]:
    with open(mapping_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clamp(val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    return max(min_val, min(val, max_val))


def normalize_bbox(
    xc: float, yc: float, w: float, h: float, eps: float = 1e-3
) -> Optional[Tuple[float, float, float, float]]:
    """Validate and clamp normalized bounding box coordinates."""
    if w <= 0.0 or h <= 0.0:
        return None
    # Reject extreme outliers
    if xc < -eps or xc > 1.0 + eps or yc < -eps or yc > 1.0 + eps:
        return None
    
    # Calculate corner bounds
    x1 = clamp(xc - w / 2.0, 0.0, 1.0)
    x2 = clamp(xc + w / 2.0, 0.0, 1.0)
    y1 = clamp(yc - h / 2.0, 0.0, 1.0)
    y2 = clamp(yc + h / 2.0, 0.0, 1.0)
    
    clamped_w = x2 - x1
    clamped_h = y2 - y1
    if clamped_w <= 0.0 or clamped_h <= 0.0:
        return None
    
    clamped_xc = (x1 + x2) / 2.0
    clamped_yc = (y1 + y2) / 2.0
    return round(clamped_xc, 6), round(clamped_yc, 6), round(clamped_w, 6), round(clamped_h, 6)


def normalize_yolo_dataset(
    input_dir: Path,
    output_dir: Path,
    source_mapping: Dict[Any, Any],
    source_class_names: Optional[List[str]] = None,
    keep_empty: bool = True,
    copy_images: bool = True,
) -> Dict[str, Any]:
    """Normalize a dataset already formatted as YOLO txt annotations."""
    images_in = input_dir / "images" if (input_dir / "images").is_dir() else input_dir
    labels_in = input_dir / "labels" if (input_dir / "labels").is_dir() else input_dir

    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in images_in.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])
    
    total_images = len(image_files)
    total_read = 0
    total_mapped = 0
    total_dropped = 0
    class_counts: Dict[int, int] = {}
    skipped_images = 0

    for img_path in image_files:
        stem = img_path.stem
        # Look for corresponding label
        lbl_path = labels_in / f"{stem}.txt"
        if not lbl_path.is_file():
            # Alternative: search recursively for stem.txt
            matches = list(labels_in.rglob(f"{stem}.txt"))
            lbl_path = matches[0] if matches else None

        valid_lines: List[str] = []
        if lbl_path and lbl_path.is_file():
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    total_read += 1
                    raw_cls_str = parts[0]
                    
                    target_cls = None
                    # Try int key first
                    try:
                        raw_cls_int = int(raw_cls_str)
                        if raw_cls_int in source_mapping:
                            target_cls = source_mapping[raw_cls_int]
                        elif str(raw_cls_int) in source_mapping:
                            target_cls = source_mapping[str(raw_cls_int)]
                        elif source_class_names and 0 <= raw_cls_int < len(source_class_names):
                            cls_name = source_class_names[raw_cls_int]
                            target_cls = source_mapping.get(cls_name, source_mapping.get(cls_name.lower()))
                    except ValueError:
                        # String class
                        target_cls = source_mapping.get(raw_cls_str, source_mapping.get(raw_cls_str.lower()))

                    if target_cls is None or not isinstance(target_cls, int) or target_cls < 0:
                        total_dropped += 1
                        continue

                    # Validate coordinates
                    try:
                        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        norm_box = normalize_bbox(xc, yc, w, h)
                        if norm_box is None:
                            total_dropped += 1
                            continue
                        n_xc, n_yc, n_w, n_h = norm_box
                        valid_lines.append(f"{target_cls} {n_xc:.6f} {n_yc:.6f} {n_w:.6f} {n_h:.6f}")
                        total_mapped += 1
                        class_counts[target_cls] = class_counts.get(target_cls, 0) + 1
                    except (IndexError, ValueError):
                        total_dropped += 1
                        continue

        if not valid_lines and not keep_empty:
            skipped_images += 1
            continue

        # Write out image and label
        dest_img = out_images / img_path.name
        if copy_images:
            if not dest_img.exists() or dest_img.resolve() != img_path.resolve():
                shutil.copy2(img_path, dest_img)
        else:
            if not dest_img.exists():
                os.symlink(img_path.resolve(), dest_img)

        dest_lbl = out_labels / f"{stem}.txt"
        with open(dest_lbl, "w", encoding="utf-8") as f:
            for l in valid_lines:
                f.write(l + "\n")

    return {
        "status": "success",
        "format": "yolo",
        "total_images_processed": total_images,
        "images_written": total_images - skipped_images,
        "skipped_empty_images": skipped_images,
        "total_annotations_read": total_read,
        "total_annotations_mapped": total_mapped,
        "total_annotations_dropped": total_dropped,
        "class_distribution": class_counts,
    }


def normalize_coco_dataset(
    coco_json_path: Path,
    images_dir: Path,
    output_dir: Path,
    source_mapping: Dict[Any, Any],
    keep_empty: bool = True,
    copy_images: bool = True,
) -> Dict[str, Any]:
    """Normalize a COCO JSON dataset to canonical YOLO format."""
    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    categories = {cat["id"]: cat.get("name", "") for cat in coco_data.get("categories", [])}
    images_meta = {img["id"]: img for img in coco_data.get("images", [])}

    # Group annotations by image_id
    img_annotations: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id is not None:
            img_annotations.setdefault(img_id, []).append(ann)

    total_images = len(images_meta)
    total_read = 0
    total_mapped = 0
    total_dropped = 0
    class_counts: Dict[int, int] = {}
    skipped_images = 0

    for img_id, img_info in images_meta.items():
        file_name = img_info.get("file_name", "")
        img_w = float(img_info.get("width", 0))
        img_h = float(img_info.get("height", 0))

        img_file_path = images_dir / file_name
        if not img_file_path.is_file():
            # Try searching by basename
            matches = list(images_dir.rglob(Path(file_name).name))
            if matches:
                img_file_path = matches[0]
            else:
                continue

        # If dimensions not in metadata, inspect file
        if img_w <= 0 or img_h <= 0:
            cv_img = cv2.imread(str(img_file_path))
            if cv_img is None:
                continue
            img_h, img_w = cv_img.shape[:2]

        anns = img_annotations.get(img_id, [])
        valid_lines: List[str] = []

        for ann in anns:
            total_read += 1
            cat_id = ann.get("category_id")
            cat_name = categories.get(cat_id, "")

            target_cls = None
            if cat_id in source_mapping:
                target_cls = source_mapping[cat_id]
            elif str(cat_id) in source_mapping:
                target_cls = source_mapping[str(cat_id)]
            elif cat_name and cat_name in source_mapping:
                target_cls = source_mapping[cat_name]
            elif cat_name and cat_name.lower() in source_mapping:
                target_cls = source_mapping[cat_name.lower()]

            if target_cls is None or not isinstance(target_cls, int) or target_cls < 0:
                total_dropped += 1
                continue

            # COCO bbox: [x_min, y_min, width, height]
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                total_dropped += 1
                continue

            bx, by, bw, bh = bbox
            if bw <= 0 or bh <= 0 or img_w <= 0 or img_h <= 0:
                total_dropped += 1
                continue

            # Convert to normalized xc, yc, w, h
            norm_xc = (bx + bw / 2.0) / img_w
            norm_yc = (by + bh / 2.0) / img_h
            norm_w = bw / img_w
            norm_h = bh / img_h

            norm_box = normalize_bbox(norm_xc, norm_yc, norm_w, norm_h)
            if norm_box is None:
                total_dropped += 1
                continue

            n_xc, n_yc, n_w, n_h = norm_box
            valid_lines.append(f"{target_cls} {n_xc:.6f} {n_yc:.6f} {n_w:.6f} {n_h:.6f}")
            total_mapped += 1
            class_counts[target_cls] = class_counts.get(target_cls, 0) + 1

        if not valid_lines and not keep_empty:
            skipped_images += 1
            continue

        dest_img = out_images / Path(file_name).name
        if copy_images:
            if not dest_img.exists() or dest_img.resolve() != img_file_path.resolve():
                shutil.copy2(img_file_path, dest_img)
        else:
            if not dest_img.exists():
                os.symlink(img_file_path.resolve(), dest_img)

        dest_lbl = out_labels / f"{Path(file_name).stem}.txt"
        with open(dest_lbl, "w", encoding="utf-8") as f:
            for l in valid_lines:
                f.write(l + "\n")

    return {
        "status": "success",
        "format": "coco",
        "total_images_processed": total_images,
        "images_written": total_images - skipped_images,
        "skipped_empty_images": skipped_images,
        "total_annotations_read": total_read,
        "total_annotations_mapped": total_mapped,
        "total_annotations_dropped": total_dropped,
        "class_distribution": class_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize dataset labels to canonical YOLO taxonomy.")
    parser.add_argument("--format", choices=["yolo", "coco"], default="yolo", help="Input dataset format")
    parser.add_argument("--input-dir", type=str, help="Input directory (for YOLO format) or dataset root")
    parser.add_argument("--coco-json", type=str, help="Path to COCO JSON annotations file")
    parser.add_argument("--images-dir", type=str, help="Path to image directory (required for COCO)")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to write normalized dataset")
    parser.add_argument("--mapping", type=str, required=True, help="Mapping key in label_mapping.yaml (e.g. coco, exam_objects)")
    parser.add_argument("--label-mapping-file", type=str, default="training/config/label_mapping.yaml", help="Path to label_mapping.yaml")
    parser.add_argument("--classes-file", type=str, default=None, help="Optional classes.txt for source class index names")
    parser.add_argument("--drop-empty", action="store_true", help="Drop images that have 0 annotations after filtering")
    parser.add_argument("--symlink", action="store_true", help="Symlink images instead of copying")

    args = parser.parse_args()

    mapping_path = Path(args.label_mapping_file)
    if not mapping_path.is_file():
        print(f"Error: label mapping file not found at {mapping_path}", file=sys.stderr)
        return 1

    mapping_doc = load_label_mapping(mapping_path)
    source_mappings = mapping_doc.get("source_mappings", {})
    if args.mapping not in source_mappings:
        print(f"Error: Mapping '{args.mapping}' not found in {mapping_path}. Available: {list(source_mappings.keys())}", file=sys.stderr)
        return 1

    source_mapping = source_mappings[args.mapping]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_class_names = None
    if args.classes_file and Path(args.classes_file).is_file():
        with open(args.classes_file, "r", encoding="utf-8") as f:
            source_class_names = [line.strip() for line in f if line.strip()]

    if args.format == "yolo":
        if not args.input_dir:
            print("Error: --input-dir required for format=yolo", file=sys.stderr)
            return 1
        summary = normalize_yolo_dataset(
            input_dir=Path(args.input_dir),
            output_dir=output_dir,
            source_mapping=source_mapping,
            source_class_names=source_class_names,
            keep_empty=not args.drop_empty,
            copy_images=not args.symlink,
        )
    elif args.format == "coco":
        if not args.coco_json or not args.images_dir:
            print("Error: --coco-json and --images-dir required for format=coco", file=sys.stderr)
            return 1
        summary = normalize_coco_dataset(
            coco_json_path=Path(args.coco_json),
            images_dir=Path(args.images_dir),
            output_dir=output_dir,
            source_mapping=source_mapping,
            keep_empty=not args.drop_empty,
            copy_images=not args.symlink,
        )
    else:
        print(f"Unsupported format: {args.format}", file=sys.stderr)
        return 1

    summary_file = output_dir / "normalization_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Normalization complete.")
    print(f"  Processed images: {summary['total_images_processed']}")
    print(f"  Annotations read: {summary['total_annotations_read']}")
    print(f"  Annotations mapped: {summary['total_annotations_mapped']}")
    print(f"  Annotations dropped: {summary['total_annotations_dropped']}")
    print(f"  Class distribution: {summary['class_distribution']}")
    print(f"  Summary saved to: {summary_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
