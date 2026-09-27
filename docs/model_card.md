# Model Card: AI Exam Hall Monitoring YOLOv8 Detector

## 1. Model Details
- **Model Name:** AI Exam Hall Monitoring YOLOv8 Detector (Baseline: `yolov8n.pt`, Candidate: `models/custom/exam_yolov8n_v1`)
- **Architecture:** Ultralytics YOLOv8 Nano (anchor-free single-stage object detector)
- **Input Resolution:** $640 \times 640$ pixels (RGB)
- **Base Pretrained Weights:** MS COCO Pretrained (`yolov8n.pt`)
- **Number of Parameters:** ~3.2M parameters
- **Canonical Object Classes (8):**
  - `0: person` — Exam candidate sitting at desk
  - `1: phone` — Smartphone, mobile device
  - `2: paper` — Cheat sheet, unauthorized notes, loose paper
  - `3: book` — Textbook, notebook, bound reading material
  - `4: calculator` — Handheld scientific or basic calculator
  - `5: laptop` — Laptop, netbook, tablet
  - `6: watch` — Wristwatch, smartwatch
  - `7: earphone_or_earbud` — Wireless earbud, wired earphone

---

## 2. Intended Use
- **Primary Use Case:** Real-time visual assistance and auxiliary telemetry for proctoring human exam invigilators.
- **Operational Setting:** Controlled indoor academic exam halls and computer-based testing (CBT) workstations with consenting test-takers.
- **Workflow Role:** Advisory signal generator. All detections contribute to the multi-signal `RiskEngine` risk score and logged visual evidence frames. **Final disciplinary decisions remain strictly the responsibility of accredited human proctors.**

---

## 3. Out-of-Scope & Prohibited Misuse
- **Fully Automated Disqualification:** The model must **never** be used to automatically terminate an exam session or fail a student without manual human invigilator review.
- **Unconsented Public Surveillance:** Deployment in public non-examination spaces or without explicit institutional disclosure and examinee consent is prohibited.
- **Biometric Identity Inference:** This model does not perform demographic profiling, race/gender estimation, or emotion classification.

---

## 4. Dataset & Training Pipeline
- **Data Engineering:**
  - Standardized canonical label taxonomy with explicit mapping rules (`training/config/label_mapping.yaml`).
  - Procedural geometric scene synthesis (`training/scripts/generate_synthetic.py`) eliminating the need for human volunteer recruitment while avoiding generative deepfake artifacts.
  - Public dataset integration via legally compliant registered sources (`training/datasets/registry.yaml`) with explicit license auditing (`training/datasets/ATTRIBUTIONS.md`).
- **Data Isolation:**
  - Group-aware, source-isolated train/val/test splitting (`training/scripts/split_dataset.py`) to prevent data leakage.
  - Perceptual difference hashing (dHash) and exact SHA-256 deduplication (`training/scripts/deduplicate_dataset.py`).

---

## 5. Evaluation & Performance Standards
- **Evaluation Splits:** Dedicated held-out test splits evaluated across canonical classes.
- **Metrics Collected:** Precision, Recall, mAP@0.50, mAP@0.50:0.95, and inference latency (mean, median, p95).
- **Conservative Promotion Rule:** A candidate custom model will **not** replace the production baseline model (`weights/yolov8n.pt`) unless objective benchmark evaluation demonstrates superior performance across target exam object categories without latency regression.

---

## 6. Known Limitations
- **Occlusion:** Highly obscured miniature devices (e.g., hidden micro-earpieces covered by hair, concealed smartwatches under long cuffs) may yield false negatives.
- **Lighting Extremes:** Harsh backlighting or very low-light environments degrade edge detection fidelity.
- **Permitted Stationeries:** Calculators or formula booklets explicitly allowed for specific exam papers may trigger advisory alerts unless silenced in exam session configurations.

---

## 7. Ethical, Privacy, and Regulatory Considerations
- **Data Minimization:** No personal biometric records or video streams are stored remotely. Local evidence screenshots are encrypted and retained strictly per institutional retention policies.
- **Human-in-the-Loop:** All automated findings require human verification before action is taken.
- **Transparency:** All training configurations, split manifests, and evaluation summaries are logged deterministically.
