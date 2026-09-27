"""
Tests for Training & Dataset Engineering Pipeline
------------------------------------------------
Comprehensive unit tests covering:
- Dataset registry and attribution licenses
- Label mapping canonical taxonomy
- Quality inspection (check_dataset)
- Label normalization (normalize_dataset)
- Deduplication and dHash algorithms (deduplicate_dataset)
- Deterministic synthetic scene generator (generate_synthetic)
- Group-aware dataset splitting (split_dataset)
- Model evaluation and baseline comparison logic (evaluate)
- Latency and throughput benchmarking format (benchmark)
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import yaml

from training.scripts.benchmark import benchmark_model
from training.scripts.check_dataset import inspect_dataset
from training.scripts.deduplicate_dataset import compute_dhash, hamming_distance, scan_dataset_duplicates
from training.scripts.evaluate import compare_models
from training.scripts.generate_synthetic import SyntheticExamSceneGenerator, generate_synthetic_dataset
from training.scripts.normalize_dataset import normalize_bbox, normalize_coco_dataset, normalize_yolo_dataset
from training.scripts.split_dataset import extract_group_id, split_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==============================================================================
# 1. Dataset Registry & Attributions
# ==============================================================================

def test_registry_structure_and_licenses():
    registry_path = PROJECT_ROOT / "training" / "datasets" / "registry.yaml"
    assert registry_path.is_file(), "registry.yaml must exist"

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    assert "datasets" in registry
    datasets = registry["datasets"]
    assert len(datasets) >= 3

    for ds in datasets:
        assert "id" in ds
        assert "name" in ds
        assert "license" in ds
        assert "source_url" in ds or "download_url" in ds
        assert "commercial_use_allowed" in ds
        if not ds["commercial_use_allowed"]:
            assert "non_commercial_warning" in ds or "NC" in ds["license"]


def test_attributions_file_exists():
    attrib_path = PROJECT_ROOT / "training" / "datasets" / "ATTRIBUTIONS.md"
    assert attrib_path.is_file(), "ATTRIBUTIONS.md must exist"
    content = attrib_path.read_text(encoding="utf-8")
    assert "License" in content
    assert "COCO" in content
    assert "Attribution" in content


# ==============================================================================
# 2. Canonical Label Mapping
# ==============================================================================

def test_canonical_label_mapping():
    mapping_path = PROJECT_ROOT / "training" / "config" / "label_mapping.yaml"
    assert mapping_path.is_file(), "label_mapping.yaml must exist"

    with open(mapping_path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    obj_classes = doc.get("canonical_object_classes", {})
    assert len(obj_classes) == 8
    assert obj_classes[0] == "person"
    assert obj_classes[1] == "phone"
    assert obj_classes[2] == "paper"
    assert obj_classes[3] == "book"
    assert obj_classes[4] == "calculator"
    assert obj_classes[5] == "laptop"
    assert obj_classes[6] == "watch"
    assert obj_classes[7] == "earphone_or_earbud"

    source_mappings = doc.get("source_mappings", {})
    assert "coco" in source_mappings
    coco_map = source_mappings["coco"]
    assert coco_map[0] == 0   # person -> person
    assert coco_map[67] == 1  # cell phone -> phone
    assert coco_map[73] == 3  # book -> book
    assert coco_map[63] == 5  # laptop -> laptop


# ==============================================================================
# 3. Quality Inspector (check_dataset)
# ==============================================================================

def test_check_dataset_valid(tmp_path):
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()

    # Create dummy images and labels
    img1 = np.full((100, 100, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(img_dir / "sample1.jpg"), img1)
    (lbl_dir / "sample1.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n")

    img2 = np.full((100, 100, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(img_dir / "sample2.jpg"), img2)
    (lbl_dir / "sample2.txt").write_text("2 0.4 0.4 0.3 0.3\n")

    report = inspect_dataset(tmp_path, max_classes=8)
    assert report["total_images_found"] == 2
    assert report["valid_images"] == 2
    assert report["corrupt_images_count"] == 0
    assert report["orphan_images_count"] == 0
    assert report["orphan_labels_count"] == 0
    assert report["valid_annotations"] == 3
    assert report["class_distribution"][0] == 1
    assert report["class_distribution"][1] == 1
    assert report["class_distribution"][2] == 1


def test_check_dataset_orphans_and_invalid(tmp_path):
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()

    # Image with no label (orphan image)
    cv2.imwrite(str(img_dir / "orphan_img.jpg"), np.zeros((50, 50, 3), dtype=np.uint8))

    # Label with no image (orphan label)
    (lbl_dir / "orphan_lbl.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    # Pair with invalid coordinates
    cv2.imwrite(str(img_dir / "invalid_box.jpg"), np.zeros((50, 50, 3), dtype=np.uint8))
    (lbl_dir / "invalid_box.txt").write_text("0 1.5 0.5 0.2 0.2\n0 0.5 0.5 0.0 0.0\n")

    report = inspect_dataset(tmp_path, max_classes=8)
    assert report["orphan_images_count"] == 1
    assert report["orphan_labels_count"] == 1
    assert report["invalid_annotations_count"] == 2


# ==============================================================================
# 4. Label Normalization (normalize_dataset)
# ==============================================================================

def test_normalize_bbox():
    # Valid box
    box = normalize_bbox(0.5, 0.5, 0.2, 0.2)
    assert box is not None
    xc, yc, w, h = box
    assert xc == 0.5 and yc == 0.5 and w == 0.2 and h == 0.2

    # Zero or negative dimension
    assert normalize_bbox(0.5, 0.5, 0.0, 0.2) is None
    assert normalize_bbox(0.5, 0.5, 0.2, -0.1) is None

    # Slightly out of bounds should clamp
    clamped = normalize_bbox(1.0005, 0.5, 0.1, 0.1)
    assert clamped is not None
    assert clamped[0] <= 1.0


def test_normalize_yolo_dataset(tmp_path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    (in_dir / "images").mkdir(parents=True)
    (in_dir / "labels").mkdir(parents=True)

    cv2.imwrite(str(in_dir / "images" / "test1.jpg"), np.zeros((100, 100, 3), dtype=np.uint8))
    # Class 67 (COCO phone) should map to 1; Class 999 (unmapped) should be dropped
    (in_dir / "labels" / "test1.txt").write_text("67 0.5 0.5 0.2 0.2\n999 0.1 0.1 0.1 0.1\n")

    source_mapping = {67: 1}
    summary = normalize_yolo_dataset(in_dir, out_dir, source_mapping=source_mapping)

    assert summary["total_images_processed"] == 1
    assert summary["total_annotations_mapped"] == 1
    assert summary["total_annotations_dropped"] == 1
    assert summary["class_distribution"] == {1: 1}

    out_lbl = out_dir / "labels" / "test1.txt"
    assert out_lbl.is_file()
    lines = out_lbl.read_text().strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("1 ")


def test_normalize_coco_dataset(tmp_path):
    images_dir = tmp_path / "coco_images"
    images_dir.mkdir()
    cv2.imwrite(str(images_dir / "img01.jpg"), np.zeros((100, 200, 3), dtype=np.uint8))

    coco_json = tmp_path / "instances.json"
    coco_data = {
        "images": [{"id": 1, "file_name": "img01.jpg", "width": 200, "height": 100}],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "dog"}],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [50, 20, 40, 60]},  # Mapped
            {"id": 11, "image_id": 1, "category_id": 2, "bbox": [10, 10, 20, 20]},  # Dropped
        ],
    }
    coco_json.write_text(json.dumps(coco_data), encoding="utf-8")

    out_dir = tmp_path / "out_coco"
    source_mapping = {1: 0}  # person -> 0

    summary = normalize_coco_dataset(
        coco_json_path=coco_json,
        images_dir=images_dir,
        output_dir=out_dir,
        source_mapping=source_mapping,
    )

    assert summary["total_images_processed"] == 1
    assert summary["total_annotations_mapped"] == 1
    assert summary["total_annotations_dropped"] == 1
    assert summary["class_distribution"] == {0: 1}


# ==============================================================================
# 5. Deduplication (deduplicate_dataset)
# ==============================================================================

def test_dhash_and_hamming_distance():
    # Identical images must produce distance 0
    img1 = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    h1 = compute_dhash(img1)
    h2 = compute_dhash(img1.copy())
    assert hamming_distance(h1, h2) == 0

    # Slight pixel change has small distance
    img_mod = img1.copy()
    img_mod[10, 10] = 255 - img_mod[10, 10]
    h_mod = compute_dhash(img_mod)
    assert hamming_distance(h1, h_mod) <= 4


def test_scan_dataset_duplicates(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    img_base = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.rectangle(img_base, (10, 10), (40, 40), (255, 255, 255), -1)

    # Save exact duplicate
    cv2.imwrite(str(img_dir / "img_orig.jpg"), img_base)
    cv2.imwrite(str(img_dir / "img_exact_dup.jpg"), img_base)

    # Save distinct image
    img_distinct = np.full((64, 64, 3), 180, dtype=np.uint8)
    cv2.imwrite(str(img_dir / "img_distinct.jpg"), img_distinct)

    report = scan_dataset_duplicates(tmp_path, dhash_threshold=2)
    assert report["total_images_scanned"] == 3
    assert report["exact_duplicate_clusters_count"] == 1
    assert report["exact_duplicate_redundant_count"] == 1


# ==============================================================================
# 6. Synthetic Scene Generator (generate_synthetic)
# ==============================================================================

def test_synthetic_generator_determinism():
    gen1 = SyntheticExamSceneGenerator(seed=123, width=320, height=240)
    img1, anns1 = gen1.generate_scene()

    gen2 = SyntheticExamSceneGenerator(seed=123, width=320, height=240)
    img2, anns2 = gen2.generate_scene()

    assert np.array_equal(img1, img2), "Identical seeds must produce identical scene pixels"
    assert anns1 == anns2, "Identical seeds must produce identical annotations"


def test_synthetic_dataset_generation(tmp_path):
    out_dir = tmp_path / "synthetic"
    meta = generate_synthetic_dataset(
        output_dir=out_dir,
        count=5,
        seed=42,
        width=320,
        height=240,
        allow_empty=False,
    )

    assert meta["total_images"] == 5
    images = list((out_dir / "images").glob("*.jpg"))
    labels = list((out_dir / "labels").glob("*.txt"))
    assert len(images) == 5
    assert len(labels) == 5

    # Check bounding box validity in all labels
    for lbl in labels:
        lines = lbl.read_text().strip().splitlines()
        for line in lines:
            parts = line.split()
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            assert 0 <= cls_id <= 7
            assert 0.0 <= xc <= 1.0
            assert 0.0 <= yc <= 1.0
            assert 0.0 < w <= 1.0
            assert 0.0 < h <= 1.0


# ==============================================================================
# 7. Group-Aware Dataset Splitting (split_dataset)
# ==============================================================================

def test_extract_group_id():
    assert extract_group_id("studentA_cam1_001") == "studentA"
    assert extract_group_id("session2_frame45") == "session2"
    assert extract_group_id("synth_000003") == "synth_chunk_0"
    assert extract_group_id("synth_000012") == "synth_chunk_2"
    assert extract_group_id("single") == "single"


def test_split_dataset(tmp_path):
    in_dir = tmp_path / "dataset_in"
    out_dir = tmp_path / "dataset_out"
    (in_dir / "images").mkdir(parents=True)
    (in_dir / "labels").mkdir(parents=True)

    # Create 10 images across 2 distinct groups
    for i in range(5):
        cv2.imwrite(str(in_dir / "images" / f"grpA_{i:03d}.jpg"), np.zeros((50, 50, 3), dtype=np.uint8))
        (in_dir / "labels" / f"grpA_{i:03d}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    for i in range(5):
        cv2.imwrite(str(in_dir / "images" / f"grpB_{i:03d}.jpg"), np.zeros((50, 50, 3), dtype=np.uint8))
        (in_dir / "labels" / f"grpB_{i:03d}.txt").write_text("1 0.4 0.4 0.1 0.1\n")

    manifest = split_dataset(
        input_dir=in_dir,
        output_dir=out_dir,
        train_ratio=0.5,
        val_ratio=0.5,
        test_ratio=0.0,
        seed=42,
    )

    assert manifest["total_images"] == 10
    file_map = manifest["file_split_map"]

    # Verify group isolation: all grpA must be in the same split
    grpA_splits = {split for name, split in file_map.items() if name.startswith("grpA")}
    assert len(grpA_splits) == 1, "Group A samples must not be fragmented across splits"

    # Verify dataset.yaml generated
    yaml_path = out_dir / "dataset.yaml"
    assert yaml_path.is_file()
    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    assert "train" in data_cfg
    assert "val" in data_cfg
    assert "names" in data_cfg


# ==============================================================================
# 8. Model Evaluation & Comparison Logic (evaluate)
# ==============================================================================

def test_compare_models_promotion_logic():
    # Scenario 1: Candidate significantly better
    cand_better = {"model_path": "cand.pt", "mAP50": 0.85, "mAP50_95": 0.65, "precision": 0.80, "recall": 0.82}
    base = {"model_path": "base.pt", "mAP50": 0.75, "mAP50_95": 0.55, "precision": 0.78, "recall": 0.72}

    res = compare_models(cand_better, base)
    assert res["recommendation"] == "PROMOTE_CANDIDATE"
    assert res["deltas"]["mAP50_delta"] == pytest.approx(0.10)

    # Scenario 2: Candidate worse
    cand_worse = {"model_path": "cand.pt", "mAP50": 0.60, "mAP50_95": 0.45, "precision": 0.65, "recall": 0.60}
    res_worse = compare_models(cand_worse, base)
    assert res_worse["recommendation"] == "RETAIN_BASELINE"


# ==============================================================================
# 9. Latency Benchmarking (benchmark)
# ==============================================================================

def test_benchmark_metrics_format():
    # Mock Ultralytics YOLO to avoid real inference overhead in unit test
    with patch("ultralytics.YOLO") as mock_yolo_cls:
        mock_instance = MagicMock()
        mock_instance.predict.return_value = []
        mock_yolo_cls.return_value = mock_instance

        results = benchmark_model(
            model_path="dummy_model.pt",
            imgsz=320,
            iterations=5,
            warmup=2,
            device="cpu",
        )

        assert "metrics" in results
        m = results["metrics"]
        assert "mean_latency_ms" in m
        assert "median_latency_ms" in m
        assert "p95_latency_ms" in m
        assert "average_fps" in m
        assert m["mean_latency_ms"] >= 0
