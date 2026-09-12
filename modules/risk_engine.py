"""
Risk Engine
-----------
Maintains and updates the risk score for every student.

Author: AI Exam Hall Monitoring System
"""

from dataclasses import dataclass, field
from typing import Dict
import time


@dataclass
class StudentRisk:

    student_id: int

    risk_score: float = 0.0

    warning_level: str = "NORMAL"

    last_activity: str = "Normal"

    last_update: float = field(
        default_factory=time.time
    )

    violations: int = 0


class RiskEngine:

    MAX_RISK = 100
    MIN_RISK = 0

    def __init__(self):

        self.students: Dict[int, StudentRisk] = {}

        # Risk points for each suspicious activity
        self.risk_table = {

            # Gaze events
            "LOOK_LEFT": 8,
            "LOOK_RIGHT": 8,
            "LOOK_UP": 6,
            "LOOK_DOWN": 5,
            "LOOKING_AWAY": 10,

            # Hand behaviour
            "HAND_FACE": 12,
            "HAND_RAISED": 10,

            # Body behaviour
            "LEANING": 10,
            "STANDING": 25,

            # Student monitoring
            "STUDENT_ABSENT": 30,
            "MULTIPLE_PERSON": 40,

            # Forbidden objects
            "CELL_PHONE": 60,
            "BOOK": 30,
            "BOOK_DETECTED": 30,
            "LAPTOP": 50,
            "LAPTOP_DETECTED": 50,

            # Audio behaviour
            "TALKING": 20,

            # Normal behaviour reduces risk
            "NORMAL": -2
        }

    # ==========================================
    # CREATE STUDENT
    # ==========================================

    def create_student(self, student_id: int):

        if student_id not in self.students:

            self.students[student_id] = StudentRisk(
                student_id=student_id
            )

        return self.students[student_id]

    # ==========================================
    # UPDATE RISK
    # ==========================================

    def update(self, student_id: int, activity: str):

        student = self.create_student(student_id)

        activity = activity.upper()

        points = self.risk_table.get(
            activity,
            0
        )

        now = time.time()

        # ------------------------------------------
        # NORMAL BEHAVIOUR
        # ------------------------------------------

        if activity == "NORMAL":

            student.risk_score = max(
                self.MIN_RISK,
                student.risk_score + points
            )

            student.last_activity = activity

            student.last_update = now

            student.warning_level = self.calculate_level(
                student.risk_score
            )

            return student

        # ------------------------------------------
        # SUSPICIOUS BEHAVIOUR
        # ------------------------------------------

        if points > 0:

            student.risk_score = min(
                student.risk_score + points,
                self.MAX_RISK
            )

            student.violations += 1

        student.last_activity = activity

        student.last_update = now

        student.warning_level = self.calculate_level(
            student.risk_score
        )

        return student

    # ==========================================
    # CALCULATE WARNING LEVEL
    # ==========================================

    def calculate_level(self, score):

        if score < 20:
            return "NORMAL"

        elif score < 40:
            return "LOW"

        elif score < 60:
            return "MEDIUM"

        elif score < 80:
            return "HIGH"

        return "CRITICAL"

    # ==========================================
    # GET STUDENT
    # ==========================================

    def get_student(self, student_id):

        return self.students.get(student_id)

    # ==========================================
    # GET ALL STUDENTS
    # ==========================================

    def get_all_students(self):

        return self.students

    # ==========================================
    # RESET STUDENT
    # ==========================================

    def reset(self, student_id):

        if student_id in self.students:

            student = self.students[student_id]

            student.risk_score = 0.0

            student.violations = 0

            student.warning_level = "NORMAL"

            student.last_activity = "NORMAL"

            student.last_update = time.time()

    # ==========================================
    # RISK DECAY
    # ==========================================

    def decay(self):

        """
        Reduces risk over time when no suspicious
        activity is detected.
        """

        now = time.time()

        for student in self.students.values():

            elapsed = now - student.last_update

            if elapsed > 5:

                student.risk_score = max(
                    self.MIN_RISK,
                    student.risk_score - 1
                )

                student.warning_level = self.calculate_level(
                    student.risk_score
                )

                student.last_update = now