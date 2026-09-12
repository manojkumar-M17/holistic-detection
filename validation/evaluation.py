"""Deterministic event matching, metrics, stability, and risk analysis."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_EVENT_TOLERANCE_SECONDS = 2.0
SUPPORTED_EVENTS = (
    "LOOK_LEFT", "LOOK_RIGHT", "LOOK_UP", "LOOK_DOWN",
    "CELL_PHONE", "BOOK_DETECTED", "LAPTOP_DETECTED",
    "HAND_FACE", "HAND_RAISED", "STANDING", "MULTIPLE_PERSON",
    "TALKING", "STUDENT_ABSENT", "LEANING",
)


@dataclass(frozen=True)
class ExpectedEvent:
    scenario_id: str
    start_time: float
    end_time: float
    expected_events: tuple[str, ...]
    expected_student_count: int | None = None
    notes: str = ""
    video_file: str | None = None
    student_id: Any = None

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "ExpectedEvent":
        if not isinstance(item, dict):
            raise ValueError("Each annotation must be an object")
        required = {"scenario_id", "start_time", "end_time", "expected_events"}
        missing = required.difference(item)
        if missing:
            raise ValueError(f"Annotation is missing fields: {sorted(missing)}")
        start = float(item["start_time"])
        end = float(item["end_time"])
        if start < 0 or end < start:
            raise ValueError("Annotation time window is invalid")
        if not isinstance(item["expected_events"], list):
            raise ValueError("expected_events must be a list")
        events = tuple(str(event).upper() for event in item["expected_events"])
        if any(not event for event in events):
            raise ValueError("expected_events cannot contain empty values")
        student_count = item.get("expected_student_count")
        return cls(
            scenario_id=str(item["scenario_id"]),
            start_time=start,
            end_time=end,
            expected_events=events,
            expected_student_count=(
                int(student_count) if student_count is not None else None
            ),
            notes=str(item.get("notes", "")),
            video_file=item.get("video_file"),
            student_id=item.get("student_id"),
        )


def load_annotations(source: dict[str, Any] | list[dict[str, Any]]) -> list[ExpectedEvent]:
    """Load and validate annotation dictionaries without requiring video files."""
    items = source.get("annotations", []) if isinstance(source, dict) else source
    if not isinstance(items, list):
        raise ValueError("Annotations must be a list")
    return [ExpectedEvent.from_dict(item) for item in items]


def load_annotation_file(path: str | Path) -> list[ExpectedEvent]:
    """Load annotations from JSON and normalize file errors for CLI callers."""
    try:
        source = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Annotation file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Annotation file is not valid JSON: {path}") from exc
    return load_annotations(source)


def _event_name(detection: dict[str, Any]) -> str:
    return str(detection.get("event", "")).upper()


def _timestamp(detection: dict[str, Any]) -> float:
    return float(detection.get("timestamp", 0.0))


def _window_contains(annotation: ExpectedEvent, timestamp: float, tolerance: float) -> bool:
    return annotation.start_time - tolerance <= timestamp <= annotation.end_time + tolerance


def evaluate_events(
    annotations: Iterable[ExpectedEvent],
    detections: Iterable[dict[str, Any]],
    tolerance_seconds: float = DEFAULT_EVENT_TOLERANCE_SECONDS,
) -> dict[str, Any]:
    """Match each expected event to at most one detection within its time window."""
    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds must be non-negative")

    expected = list(annotations)
    observed = [item for item in detections if _event_name(item)]
    matched_detection_indexes: set[int] = set()
    matched_expected_indexes: set[int] = set()
    true_positives: list[dict[str, Any]] = []
    false_negatives: list[dict[str, Any]] = []

    for expected_index, annotation in enumerate(expected):
        for event in annotation.expected_events:
            candidates = [
                (index, detection)
                for index, detection in enumerate(observed)
                if index not in matched_detection_indexes
                and _event_name(detection) == event
                and _window_contains(annotation, _timestamp(detection), tolerance_seconds)
            ]
            if candidates:
                index, detection = min(
                    candidates,
                    key=lambda pair: abs(_timestamp(pair[1]) - annotation.start_time),
                )
                matched_detection_indexes.add(index)
                matched_expected_indexes.add(expected_index)
                true_positives.append(
                    {
                        "event": event,
                        "scenario_id": annotation.scenario_id,
                        "expected_window": [annotation.start_time, annotation.end_time],
                        "detection": detection,
                    }
                )
            else:
                false_negatives.append(
                    {
                        "event": event,
                        "scenario_id": annotation.scenario_id,
                        "expected_window": [annotation.start_time, annotation.end_time],
                        "related_detections": [
                            detection
                            for detection in observed
                            if _event_name(detection) == event
                        ],
                    }
                )

    false_positives = [
        detection
        for index, detection in enumerate(observed)
        if index not in matched_detection_indexes
    ]

    expected_events = [
        event
        for annotation in expected
        for event in annotation.expected_events
    ]
    detected_events = [_event_name(item) for item in observed]
    event_types = sorted(set(expected_events).union(detected_events))
    metrics: dict[str, dict[str, Any]] = {}
    for event in sorted(set(event_types).union(SUPPORTED_EVENTS)):
        tp = sum(item["event"] == event for item in true_positives)
        fn = sum(item["event"] == event for item in false_negatives)
        fp = sum(_event_name(item) == event for item in false_positives)
        metrics[event] = calculate_metrics(tp, fp, fn)
        if not any(event in annotation.expected_events for annotation in expected) and not any(
            _event_name(item) == event for item in observed
        ):
            metrics[event]["status"] = "insufficient_data"

    return {
        "tolerance_seconds": tolerance_seconds,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "metrics": metrics,
        "overall": calculate_metrics(
            len(true_positives), len(false_positives), len(false_negatives)
        ),
    }


def calculate_metrics(true_positives: int, false_positives: int, false_negatives: int) -> dict[str, Any]:
    """Calculate precision, recall, and F1 with explicit unavailable states."""
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = (
        true_positives / precision_denominator if precision_denominator else None
    )
    recall = true_positives / recall_denominator if recall_denominator else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "status": "not_applicable" if precision is None and recall is None else "measured",
    }


def summarize_stability(detections: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize repeated event detections and transitions per student/event."""
    grouped: dict[tuple[Any, str], list[dict[str, Any]]] = defaultdict(list)
    for detection in detections:
        event = _event_name(detection)
        if event:
            grouped[(detection.get("student_id"), event)].append(detection)

    summaries = []
    for (student_id, event), items in sorted(grouped.items(), key=str):
        items.sort(key=_timestamp)
        timestamps = [_timestamp(item) for item in items]
        risks = [float(item["risk_score"]) for item in items if item.get("risk_score") is not None]
        transitions = sum(
            1
            for previous, current in zip(items, items[1:])
            if _event_name(previous) != _event_name(current)
        )
        summaries.append(
            {
                "student_id": student_id,
                "event": event,
                "detection_count": len(items),
                "first_detection": timestamps[0],
                "last_detection": timestamps[-1],
                "duration_seconds": timestamps[-1] - timestamps[0],
                "state_transitions": transitions,
                "average_risk_score": sum(risks) / len(risks) if risks else None,
                "maximum_risk_score": max(risks) if risks else None,
            }
        )
    return summaries


def analyze_event_stability(
    annotations: Iterable[ExpectedEvent],
    detections: Iterable[dict[str, Any]],
    tolerance_seconds: float = DEFAULT_EVENT_TOLERANCE_SECONDS,
) -> list[dict[str, Any]]:
    """Measure latency, coverage, interruptions, and longest runs per annotation."""
    observed = list(detections)
    summaries = []
    for annotation in annotations:
        duration = annotation.end_time - annotation.start_time
        for event in annotation.expected_events:
            matches = [
                item for item in observed
                if _event_name(item) == event
                and _window_contains(annotation, _timestamp(item), tolerance_seconds)
                and (
                    annotation.student_id is None
                    or item.get("student_id") == annotation.student_id
                )
            ]
            matches.sort(key=_timestamp)
            timestamps = [_timestamp(item) for item in matches]
            if not timestamps:
                summaries.append({
                    "scenario_id": annotation.scenario_id,
                    "event": event,
                    "detection_count": 0,
                    "detection_latency_seconds": None,
                    "coverage_ratio": 0.0,
                    "interruptions": None,
                    "longest_continuous_seconds": 0.0,
                    "status": "not_detected",
                })
                continue
            gaps = [current - previous for previous, current in zip(timestamps, timestamps[1:])]
            cadence = sorted(gaps)[(len(gaps) - 1) // 2] if gaps else 0.0
            continuity_gap = max(1.0, cadence * 2) if cadence else 1.0
            continuous_runs = []
            run_start = timestamps[0]
            previous = timestamps[0]
            for timestamp in timestamps[1:]:
                if timestamp - previous > continuity_gap:
                    continuous_runs.append(previous - run_start)
                    run_start = timestamp
                previous = timestamp
            continuous_runs.append(previous - run_start)
            detected_span = min(duration, max(0.0, timestamps[-1] - timestamps[0]))
            summaries.append({
                "scenario_id": annotation.scenario_id,
                "event": event,
                "detection_count": len(matches),
                "detection_latency_seconds": timestamps[0] - annotation.start_time,
                "coverage_ratio": detected_span / duration if duration else 1.0,
                "interruptions": max(0, len(continuous_runs) - 1),
                "longest_continuous_seconds": max(continuous_runs),
                "status": "measured",
            })
    return summaries


def summarize_risk(detections: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize risk and severity distributions from recorded detections."""
    items = list(detections)
    risks = [float(item["risk_score"]) for item in items if item.get("risk_score") is not None]
    severity_counts = Counter(
        str(item.get("severity", "UNKNOWN")) for item in items
    )
    return {
        "minimum": min(risks) if risks else None,
        "maximum": max(risks) if risks else None,
        "average": sum(risks) / len(risks) if risks else None,
        "final": risks[-1] if risks else None,
        "severity_distribution": dict(severity_counts),
        "status": "measured" if risks else "not_applicable",
    }


def compare_experiments(experiments: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a comparable table from baseline/candidate evaluation results."""
    rows = []
    for experiment in experiments:
        evaluation = experiment.get("evaluation", {})
        overall = evaluation.get("overall", {})
        configuration = experiment.get("configuration", {})
        rows.append({
            "name": experiment.get("name", experiment.get("session", "unknown")),
            "parameter_changes": experiment.get("parameter_changes", {}),
            "configuration": configuration,
            "precision": overall.get("precision"),
            "recall": overall.get("recall"),
            "f1": overall.get("f1"),
            "false_positives": overall.get("false_positives"),
            "false_negatives": overall.get("false_negatives"),
            "status": overall.get("status", "insufficient_data"),
        })
    return rows


def summarize_scenarios(
    annotations: Iterable[ExpectedEvent],
    detections: Iterable[dict[str, Any]],
    tolerance_seconds: float = DEFAULT_EVENT_TOLERANCE_SECONDS,
) -> list[dict[str, Any]]:
    """Summarize risk and severity for each annotated scenario window."""
    observed = list(detections)
    summaries = []
    for annotation in annotations:
        items = [
            item for item in observed
            if _window_contains(annotation, _timestamp(item), tolerance_seconds)
            and (
                annotation.student_id is None
                or item.get("student_id") == annotation.student_id
            )
        ]
        risks = [float(item["risk_score"]) for item in items if item.get("risk_score") is not None]
        severities = [str(item.get("severity", "UNKNOWN")) for item in items]
        severity_order = {"NORMAL": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        peak = max(severities, key=lambda value: severity_order.get(value, -1), default=None)
        summaries.append({
            "scenario_id": annotation.scenario_id,
            "event_count": len(items),
            "minimum_risk": min(risks) if risks else None,
            "maximum_risk": max(risks) if risks else None,
            "average_risk": sum(risks) / len(risks) if risks else None,
            "peak_severity": peak,
            "status": "measured" if risks else "insufficient_data",
        })
    return summaries
