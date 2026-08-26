"""
Risk Engine
------------
Maintains and updates the risk score for every student.

Author : AI Exam Hall Monitoring System
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

    last_update: float = field(default_factory=time.time)

    violations: int = 0


class RiskEngine:

    MAX_RISK = 100
    MIN_RISK = 0

    def __init__(self):

        self.students: Dict[int, StudentRisk] = {}

        self.risk_table = {

            "LOOK_LEFT": 8,
            "LOOK_RIGHT": 8,
            "LOOK_UP": 6,
            "LOOK_DOWN": 5,

            "HAND_FACE": 12,
            "HAND_RAISED": 10,

            "LEANING": 10,

            "STANDING": 25,

            "MULTIPLE_PERSON": 40,

            "CELL_PHONE": 60,

            "BOOK": 30,

            "LAPTOP": 50,

            "TALKING": 20,

            "NORMAL": -2

        }

    def create_student(self, student_id: int):

        if student_id not in self.students:

            self.students[student_id] = StudentRisk(student_id)

    def update(self, student_id: int, activity: str):

        self.create_student(student_id)

        student = self.students[student_id]

        points = self.risk_table.get(activity.upper(), 0)

        student.risk_score += points

        student.risk_score = max(
            self.MIN_RISK,
            min(student.risk_score, self.MAX_RISK)
        )

        student.last_activity = activity

        student.last_update = time.time()

        if points > 0:
            student.violations += 1

        student.warning_level = self.calculate_level(student.risk_score)

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

    def get_student(self, student_id):

        return self.students.get(student_id)

    def get_all_students(self):

        return self.students

    def reset(self, student_id):

        if student_id in self.students:

            self.students[student_id].risk_score = 0

            self.students[student_id].violations = 0

            self.students[student_id].warning_level = "NORMAL"

    def decay(self):

        """
        Reduce risk over time if the student behaves normally.
        """

        now = time.time()

        for student in self.students.values():

            elapsed = now - student.last_update

            if elapsed > 5:

                student.risk_score -= 1

                student.risk_score = max(
                    self.MIN_RISK,
                    student.risk_score
                )

                student.warning_level = self.calculate_level(
                    student.risk_score
                )

                student.last_update = now