"""
Student Manager
---------------
Maintains the current state of every tracked student.

This module connects:
    Object Tracker
        ↓
    Student Manager
        ↓
    Risk Engine
        ↓
    Dashboard / Timeline / Reports
"""

import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple


@dataclass
class StudentState:
    """Stores the current state of one student."""

    student_id: int

    bbox: Optional[Tuple[int, int, int, int]] = None

    confidence: float = 0.0

    risk_score: float = 0.0

    risk_level: str = "NORMAL"

    current_activity: str = "NORMAL"

    status: str = "ACTIVE"

    violations: int = 0

    first_seen: float = 0.0

    last_seen: float = 0.0


class StudentManager:

    def __init__(self, timeout=5):

        self.students: Dict[int, StudentState] = {}

        self.timeout = timeout

    # --------------------------------------------------
    # CREATE / REGISTER STUDENT
    # --------------------------------------------------

    def register_student(
        self,
        student_id: int,
        bbox=None,
        confidence=0.0
    ):

        current_time = time.time()

        if student_id not in self.students:

            self.students[student_id] = StudentState(

                student_id=student_id,

                bbox=bbox,

                confidence=confidence,

                first_seen=current_time,

                last_seen=current_time

            )

        else:

            self.update_student(
                student_id,
                bbox,
                confidence
            )

        return self.students[student_id]

    # --------------------------------------------------
    # UPDATE STUDENT
    # --------------------------------------------------

    def update_student(
        self,
        student_id: int,
        bbox=None,
        confidence=None
    ):

        if student_id not in self.students:

            return self.register_student(
                student_id,
                bbox,
                confidence or 0.0
            )

        student = self.students[student_id]

        if bbox is not None:
            student.bbox = bbox

        if confidence is not None:
            student.confidence = confidence

        student.last_seen = time.time()

        student.status = "ACTIVE"

        return student

    # --------------------------------------------------
    # UPDATE RISK
    # --------------------------------------------------

    def update_risk(
        self,
        student_id: int,
        risk_score: float,
        risk_level: str
    ):

        if student_id not in self.students:
            self.register_student(student_id)

        student = self.students[student_id]

        student.risk_score = max(
            0.0,
            min(100.0, float(risk_score))
        )

        student.risk_level = risk_level

        return student

    # --------------------------------------------------
    # UPDATE ACTIVITY
    # --------------------------------------------------

    def update_activity(
        self,
        student_id: int,
        activity: str
    ):

        if student_id not in self.students:
            self.register_student(student_id)

        student = self.students[student_id]

        student.current_activity = activity

        if activity != "NORMAL":
            student.violations += 1

        student.last_seen = time.time()

        return student

    # --------------------------------------------------
    # GET STUDENT
    # --------------------------------------------------

    def get_student(
        self,
        student_id: int
    ):

        return self.students.get(student_id)

    # --------------------------------------------------
    # GET ALL STUDENTS
    # --------------------------------------------------

    def get_all_students(self):

        return list(self.students.values())

    # --------------------------------------------------
    # GET ACTIVE STUDENTS
    # --------------------------------------------------

    def get_active_students(self):

        self.update_status()

        return [
            student
            for student in self.students.values()
            if student.status == "ACTIVE"
        ]

    # --------------------------------------------------
    # UPDATE ACTIVE / INACTIVE STATUS
    # --------------------------------------------------

    def update_status(self):

        current_time = time.time()

        for student in self.students.values():

            elapsed = current_time - student.last_seen

            if elapsed > self.timeout:

                student.status = "INACTIVE"

    # --------------------------------------------------
    # REMOVE STUDENT
    # --------------------------------------------------

    def remove_student(
        self,
        student_id: int
    ):

        if student_id in self.students:

            del self.students[student_id]

            return True

        return False

    # --------------------------------------------------
    # CLEAR ALL STUDENTS
    # --------------------------------------------------

    def clear(self):

        self.students.clear()

    # --------------------------------------------------
    # CONVERT TO DICTIONARY
    # --------------------------------------------------

    def get_student_dict(
        self,
        student_id: int
    ):

        student = self.get_student(student_id)

        if student is None:
            return None

        return asdict(student)

    # --------------------------------------------------
    # DASHBOARD DATA
    # --------------------------------------------------

    def get_dashboard_data(self):

        self.update_status()

        return {

            "total_students": len(self.students),

            "active_students": len(
                self.get_active_students()
            ),

            "students": [
                asdict(student)
                for student in self.students.values()
            ]

        }