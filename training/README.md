# AI Exam Hall Monitoring — Training & Dataset Engineering Pipeline

This directory contains the end-to-end reproducible dataset engineering, synthetic generation, validation, training, and benchmarking pipeline for custom YOLO models.

---

## 1. Directory Structure

```
training/
├── baselines/            # Frozen environment and test baselines
│   └── phase6_baseline.json
├── config/               # Canonical taxonomies and training hyperparameters
│   ├── label_mapping.yaml
│   └── training.yaml
├── datasets/             # Dataset registry and storage (large files ignored by git)
│   ├── registry.yaml     # Catalog of approved public datasets & licenses
│   ├── ATTRIBUTIONS.md   # Legal attribution and terms of use
│   ├── raw/              # Raw downloaded datasets (gitignored)
│   ├── normalized/       # Re-labeled canonical datasets (gitignored)
│   ├── synthetic/        # Procedurally generated exam scenes (gitignored)
│   └── processed/        # Split train/val/test datasets with dataset.yaml (gitignored)
├── reports/              # Inspection, deduplication, and comparison reports
└── scripts/              # Pipeline automation CLI scripts
    ├── check_dataset.py       # Quality inspection & integrity audit
    ├── normalize_dataset.py   # Label taxonomy remapping
    ├── deduplicate_dataset.py # Exact SHA-256 and perceptual dHash deduplication
    ├── generate_synthetic.py  # Deterministic procedural exam scene generation
    ├── split_dataset.py       # Group-aware train/val/test splitting
    ├── export_demo_frames.py  # Demo frame extraction
    ├── train.py               # Ultralytics YOLO training CLI & packaging
    ├── evaluate.py            # Candidate vs. baseline model comparison
    └── benchmark.py           # Inference latency & throughput benchmarking
```

---

## 2. Canonical Object Taxonomy

All incoming datasets and synthetic generators map to 8 standardized classes:

| Class ID | Class Name | Description |
|:---:|:---|:---|
| `0` | `person` | Student/candidate seated at exam desk |
| `1` | `phone` | Smartphone or mobile device |
| `2` | `paper` | Loose notes, cheat sheet, unauthorized paper |
| `3` | `book` | Bound notebook, textbook |
| `4` | `calculator` | Scientific or handheld calculator |
| `5` | `laptop` | Open or closed laptop/tablet |
| `6` | `watch` | Wristwatch or smartwatch |
| `7` | `earphone_or_earbud` | Wireless earbud or in-ear headphones |

See `training/config/label_mapping.yaml` for complete source-to-canonical mapping rules.

---

## 3. End-to-End Workflow

### Step 1: Generate Procedural Synthetic Dataset
Generates deterministic exam scenes with student silhouettes, desk textures, and suspicious objects:
```bash
python training/scripts/generate_synthetic.py --output-dir training/datasets/synthetic --count 100 --seed 42
```

### Step 2: Quality Inspection & Integrity Audit
Scans bounding box boundaries, orphan files, and class balances:
```bash
python training/scripts/check_dataset.py --dataset-dir training/datasets/synthetic
```
Outputs `dataset_quality_report.json` and `dataset_quality_report.html`.

### Step 3: Deduplication
Ensures no exact or near-duplicate frames leak between splits:
```bash
python training/scripts/deduplicate_dataset.py --dataset-dir training/datasets/synthetic --dhash-threshold 4 --action report
```

### Step 4: Group-Aware Dataset Splitting
Divides the dataset into 70% train, 15% val, and 15% test splits with group isolation:
```bash
python training/scripts/split_dataset.py \
  --input-dir training/datasets/synthetic \
  --output-dir training/datasets/processed/exam_dataset_v1 \
  --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 --seed 42
```
Outputs `dataset.yaml` and `split_manifest.json`.

### Step 5: Model Training
#### Fast CPU Smoke Test:
```bash
python training/scripts/train.py \
  --data training/datasets/processed/exam_dataset_v1/dataset.yaml \
  --name exam_yolov8n_smoke \
  --smoke-test
```

#### Full Training Run:
```bash
python training/scripts/train.py \
  --data training/datasets/processed/exam_dataset_v1/dataset.yaml \
  --epochs 50 \
  --batch 16 \
  --imgsz 640 \
  --name exam_yolov8n_v1
```
Trained weights and metadata are packaged into `models/custom/<name>/`.

### Step 6: Candidate Evaluation & Baseline Comparison
Compares the candidate custom model against the production baseline:
```bash
python training/scripts/evaluate.py \
  --candidate-model models/custom/exam_yolov8n_v1/best.pt \
  --baseline-model weights/yolov8n.pt \
  --data training/datasets/processed/exam_dataset_v1/dataset.yaml \
  --output-dir training/reports/model_comparison
```
Generates `model_comparison.json` and `model_comparison.html` with objective promotion recommendations.

### Step 7: Latency & Throughput Benchmarking
Measures mean, median, p95 latency and FPS:
```bash
python training/scripts/benchmark.py \
  --model models/custom/exam_yolov8n_v1/best.pt \
  --device cpu \
  --iterations 50
```

---

## 4. Policy & Promotion Safeguards

1. **No Volunteer Dependency:** Uses procedural synthesis and legally verified public data.
2. **Conservative Deployment:** The production baseline (`weights/yolov8n.pt`) remains active by default. Candidate models require objective superiority in evaluation before being promoted.
3. **Storage Hygiene:** Raw image directories and large binary weights (`*.pt`) are strictly excluded from git tracking via `.gitignore`.
