"""
Timeline Manager
----------------
Records chronological activities for every student.
"""

import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass
class TimelineEvent:
    student_id: int
    activity: str
    risk_score: float
    risk_level: str
    timestamp: float
    time_string: str


class TimelineManager:

    def __init__(self):

        self.timelines: Dict[int, List[TimelineEvent]] = {}

    # --------------------------------------------------
    # ADD EVENT
    # --------------------------------------------------

    def add_event(
        self,
        student_id: int,
        activity: str,
        risk_score: float = 0.0,
        risk_level: str = "NORMAL"
    ):

        timestamp = time.time()

        event = TimelineEvent(
            student_id=student_id,
            activity=activity,
            risk_score=float(risk_score),
            risk_level=risk_level,
            timestamp=timestamp,
            time_string=datetime.fromtimestamp(
                timestamp
            ).strftime("%Y-%m-%d %H:%M:%S")
        )

        if student_id not in self.timelines:

            self.timelines[student_id] = []

        self.timelines[student_id].append(event)

        return event

    # --------------------------------------------------
    # GET STUDENT TIMELINE
    # --------------------------------------------------

    def get_student_timeline(self, student_id):

        return self.timelines.get(student_id, [])

    # --------------------------------------------------
    # GET ALL TIMELINES
    # --------------------------------------------------

    def get_all_timelines(self):

        return self.timelines

    # --------------------------------------------------
    # GET RECENT EVENTS
    # --------------------------------------------------

    def get_recent_events(
        self,
        student_id,
        limit=10
    ):

        events = self.get_student_timeline(student_id)

        return events[-limit:]

    # --------------------------------------------------
    # COUNT EVENTS
    # --------------------------------------------------

    def get_event_count(self, student_id):

        return len(
            self.get_student_timeline(student_id)
        )

    # --------------------------------------------------
    # CLEAR STUDENT TIMELINE
    # --------------------------------------------------

    def clear_student(self, student_id):

        if student_id in self.timelines:

            del self.timelines[student_id]

    # --------------------------------------------------
    # CLEAR ALL
    # --------------------------------------------------

    def clear_all(self):

        self.timelines.clear()

    # --------------------------------------------------
    # EXPORT STUDENT TIMELINE
    # --------------------------------------------------

    def export_student(self, student_id):

        events = self.get_student_timeline(student_id)

        return [
            asdict(event)
            for event in events
        ]

    # --------------------------------------------------
    # EXPORT EVERYTHING
    # --------------------------------------------------

    def export_all(self):

        return {

            student_id: [
                asdict(event)
                for event in events
            ]

            for student_id, events
            in self.timelines.items()

        }