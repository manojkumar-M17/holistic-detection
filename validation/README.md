# Real-World Validation

This directory measures detections from local recordings. It does not upload video or claim model accuracy without ground-truth annotations.

## Layout

- `scenarios/scenarios.json`: standard scenario definitions.
- `annotations/annotations.template.json`: annotation format template.
- `recordings/`: local MP4, AVI, or MOV files. Do not commit recordings.
- `results/`: JSON results and HTML reports. Do not commit generated results.
- `reports/`: optional copied validation reports. Do not commit generated reports.
- `generated/`: temporary validation media or derived artifacts. Do not commit them.

## Controlled Recording Workflow

Record a local 5–15 minute session with one consenting participant and a fixed camera. Place the camera at approximately face height, keep the full upper body and desk visible, use stable front lighting, and avoid strong backlight or reflective backgrounds. Keep the participant seated at a consistent distance throughout the session.

Perform each scenario for roughly 5–10 seconds, return to `NORMAL_SITTING` for at least 5 seconds between scenarios, and avoid transitions while annotating. Recommended order: normal sitting, each gaze direction, phone, book, laptop, hand near face, hand raised, standing, multiple person, absence, talking, return to normal, and simultaneous events. Repeat difficult scenarios if the action was ambiguous or partly occluded.

Obtain informed consent for controlled testing. Keep the recording on the local machine, remove unnecessary personal information, anonymize where practical, delete it after evaluation, and never commit the recording or screenshots to Git.

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

Malformed or missing annotation JSON is rejected with a clear error. Missing, unreadable, unsupported, or zero-frame videos are rejected without modifying the production database.

## Metrics

Expected events are matched one-to-one to detections of the same event inside the annotation window plus a configurable default tolerance of 2 seconds. Precision, recall, and F1 use the standard TP/FP/FN formulas. When no denominator exists, the metric is reported as `null` and marked `not_applicable`, not as a misleading zero.

Stability summaries include count, first and last detection, duration, transitions, average risk, and maximum risk. Risk summaries include minimum, maximum, average, final score, and severity distribution.

Annotated runs also report detection latency, event-window coverage, interruptions, longest continuous detection, scenario risk summaries, and frame-level error context. Events with no expected or observed examples are marked `insufficient_data`, not `0%`.

## Tuning

Run a small controlled parameter grid:

```bash
python tools/tune_validation.py --video validation/recordings/session_01.mp4 --annotations validation/annotations/session_01.json --yaw-thresholds 20 25 30
```

Each run stores its configuration and result. Compare F1, false positives, false negatives, and alert frequency together; do not tune against one metric alone.

The current tuning command performs a small `HEAD_YAW_THRESHOLD` comparison. Production defaults are never changed automatically.

## Privacy and Limitations

Keep recordings local, remove identifying material where possible, and do not commit student images or video. Camera angle, lighting, occlusion, model confidence, microphone hardware, and persistence thresholds affect results. Synthetic unit tests verify the evaluator only; real-world detection metrics remain pending an annotated recording.
