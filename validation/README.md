# Real-World Validation

This directory measures detections from local recordings. It does not upload video or claim model accuracy without ground-truth annotations.

## Layout

- `scenarios/scenarios.json`: standard scenario definitions.
- `annotations/annotations.template.json`: annotation format template.
- `recordings/`: local MP4, AVI, or MOV files. Do not commit recordings.
- `results/`: JSON results and HTML reports. Do not commit generated results.

## Annotation Format

Copy the template and replace the recording path and timings:

```json
{
  "session": "session_01",
  "video_file": "recordings/session_01.mp4",
  "annotations": [
    {
      "scenario_id": "LOOK_LEFT",
      "start_time": 35.0,
      "end_time": 42.0,
      "expected_events": ["LOOK_LEFT"],
      "expected_student_count": 1,
      "notes": "Student intentionally looks left."
    }
  ]
}
```

Timings are seconds from the start of the recording. Do not invent timings before reviewing the actual recording.

## Run A Validation

From the repository root:

```bash
python tools/run_validation.py --video validation/recordings/session_01.mp4 --annotations validation/annotations/session_01.json
```

The runner calls the production `CameraManager`, `StudentDetector`, `HolisticDetector`, `analyze_student_behaviour`, and `SuspiciousEngine` components. Validation disables only incident screenshot/database writes so experiments do not contaminate production evidence.

Results are saved under `validation/results/` with the captured configuration, frame counts, processing FPS, detections, metrics, and report. Tuning runs use `validation/results/tuning/run_###/`.

## Metrics

Expected events are matched one-to-one to detections of the same event inside the annotation window plus a configurable default tolerance of 2 seconds. Precision, recall, and F1 use the standard TP/FP/FN formulas. When no denominator exists, the metric is reported as `null` and marked `not_applicable`, not as a misleading zero.

Stability summaries include count, first and last detection, duration, transitions, average risk, and maximum risk. Risk summaries include minimum, maximum, average, final score, and severity distribution.

## Tuning

Run a small controlled parameter grid:

```bash
python tools/tune_validation.py --video validation/recordings/session_01.mp4 --annotations validation/annotations/session_01.json --yaw-thresholds 20 25 30
```

Each run stores its configuration and result. Compare F1, false positives, false negatives, and alert frequency together; do not tune against one metric alone.

## Privacy and Limitations

Keep recordings local, remove identifying material where possible, and do not commit student images or video. Camera angle, lighting, occlusion, model confidence, microphone hardware, and persistence thresholds affect results. Synthetic unit tests verify the evaluator only; real-world detection metrics remain pending an annotated recording.
