"""Deterministic event matching, metrics, stability, and risk analysis."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


DEFAULT_EVENT_TOLERANCE_SECONDS = 2.0


@dataclass(frozen=True)
class ExpectedEvent:
    scenario_id: str
    start_time: float
    end_time: float
    expected_events: tuple[str, ...]
    expected_student_count: int | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "ExpectedEvent":
        required = {"scenario_id", "start_time", "end_time", "expected_events"}
        missing = required.difference(item)
        if missing:
            raise ValueError(f"Annotation is missing fields: {sorted(missing)}")
        start = float(item["start_time"])
        end = float(item["end_time"])
        if start < 0 or end < start:
            raise ValueError("Annotation time window is invalid")
        events = tuple(str(event).upper() for event in item["expected_events"])
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
        )


def load_annotations(source: dict[str, Any] | list[dict[str, Any]]) -> list[ExpectedEvent]:
    """Load and validate annotation dictionaries without requiring video files."""
    items = source.get("annotations", []) if isinstance(source, dict) else source
    if not isinstance(items, list):
        raise ValueError("Annotations must be a list")
    return [ExpectedEvent.from_dict(item) for item in items]


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
        and not any(
            _window_contains(annotation, _timestamp(detection), tolerance_seconds)
            and _event_name(detection) in annotation.expected_events
            for annotation in expected
        )
    ]

    expected_events = [
        event
        for annotation in expected
        for event in annotation.expected_events
    ]
    detected_events = [_event_name(item) for item in observed]
    event_types = sorted(set(expected_events).union(detected_events))
    metrics: dict[str, dict[str, Any]] = {}
    for event in event_types:
        tp = sum(item["event"] == event for item in true_positives)
        fn = sum(item["event"] == event for item in false_negatives)
        fp = sum(_event_name(item) == event for item in false_positives)
        metrics[event] = calculate_metrics(tp, fp, fn)

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
