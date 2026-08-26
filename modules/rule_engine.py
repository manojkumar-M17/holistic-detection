"""
Rule Engine
-----------
Evaluates student behaviour and generates events for the Risk Engine.

Author : AI Exam Hall Monitoring System
"""

from typing import Dict, List
from modules.risk_engine import RiskEngine


class RuleEngine:

    def __init__(self, risk_engine: RiskEngine):

        self.risk_engine = risk_engine

    def evaluate(self, student_id: int, behaviour: Dict) -> List[str]:
        """
        behaviour Example

        {
            "head_direction": "LEFT",
            "standing": False,
            "phone": False,
            "multiple_person": False,
            "talking": False,
            "hand_face": True,
            "leaning": False
        }
        """

        events = []

        head = behaviour.get("head_direction", "CENTER")

        if head == "LEFT":
            events.append("LOOK_LEFT")

        elif head == "RIGHT":
            events.append("LOOK_RIGHT")

        elif head == "UP":
            events.append("LOOK_UP")

        elif head == "DOWN":
            events.append("LOOK_DOWN")

        if behaviour.get("standing", False):
            events.append("STANDING")

        if behaviour.get("phone", False):
            events.append("CELL_PHONE")

        if behaviour.get("multiple_person", False):
            events.append("MULTIPLE_PERSON")

        if behaviour.get("talking", False):
            events.append("TALKING")

        if behaviour.get("hand_face", False):
            events.append("HAND_FACE")

        if behaviour.get("leaning", False):
            events.append("LEANING")

        # Normal behaviour
        if len(events) == 0:
            events.append("NORMAL")

        # Update Risk Engine
        for event in events:
            self.risk_engine.update(student_id, event)

        return events