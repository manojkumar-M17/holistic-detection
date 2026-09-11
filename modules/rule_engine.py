"""
Rule Engine
-----------
Evaluates student behaviour and generates standardized
suspicious events for the Risk Engine.

Author: AI Exam Hall Monitoring System
"""

from typing import Dict, List
from modules.risk_engine import RiskEngine


class RuleEngine:

    def __init__(self, risk_engine: RiskEngine):
        self.risk_engine = risk_engine

    def evaluate(self, student_id: int, behaviour: Dict) -> List[str]:
        """
        Evaluates behaviour features and generates events.

        Supports output from modules/behaviour.py.
        """

        events = []

        # ==========================================
        # 1. GAZE / LOOKING AWAY DETECTION
        # ==========================================

        gaze_event = behaviour.get("gaze_event")

        gaze_direction = behaviour.get(
            "gaze_direction",
            "UNKNOWN"
        )

        is_looking_away = behaviour.get(
            "is_looking_away",
            False
        )

        # Use stable gaze event first
        if gaze_event:

            if gaze_event == "LOOKING_LEFT":
                events.append("LOOK_LEFT")

            elif gaze_event == "LOOKING_RIGHT":
                events.append("LOOK_RIGHT")

            elif gaze_event == "LOOKING_UP":
                events.append("LOOK_UP")

            elif gaze_event == "LOOKING_DOWN":
                events.append("LOOK_DOWN")

            else:
                events.append("LOOKING_AWAY")

        # Fallback to direction
        elif is_looking_away:

            if gaze_direction == "LOOKING_LEFT":
                events.append("LOOK_LEFT")

            elif gaze_direction == "LOOKING_RIGHT":
                events.append("LOOK_RIGHT")

            elif gaze_direction == "LOOKING_UP":
                events.append("LOOK_UP")

            elif gaze_direction == "LOOKING_DOWN":
                events.append("LOOK_DOWN")

            else:
                events.append("LOOKING_AWAY")

        # ==========================================
        # 2. STANDING DETECTION
        # ==========================================

        if behaviour.get("is_standing", False):
            events.append("STANDING")

        # Backward compatibility
        elif behaviour.get("standing", False):
            events.append("STANDING")

        # ==========================================
        # 3. FORBIDDEN OBJECT DETECTION
        # ==========================================

        if behaviour.get("has_forbidden_object", False):

            forbidden_objects = behaviour.get(
                "forbidden_objects",
                []
            )

            for obj in forbidden_objects:

                obj_upper = obj.upper()

                if "PHONE" in obj_upper or "CELL" in obj_upper:
                    events.append("CELL_PHONE")

                elif "BOOK" in obj_upper:
                    events.append("BOOK_DETECTED")

                elif "LAPTOP" in obj_upper:
                    events.append("LAPTOP_DETECTED")

                else:
                    events.append(
                        f"FORBIDDEN_OBJECT_{obj_upper}"
                    )

        # Backward compatibility
        elif behaviour.get("phone", False):
            events.append("CELL_PHONE")

        # ==========================================
        # 4. MULTIPLE PERSON DETECTION
        # ==========================================

        if behaviour.get("multiple_person", False):
            events.append("MULTIPLE_PERSON")

        # ==========================================
        # 5. AUDIO / TALKING DETECTION
        # ==========================================

        if behaviour.get("talking", False):
            events.append("TALKING")

        # ==========================================
        # 6. HAND NEAR FACE
        # ==========================================

        if behaviour.get("hand_near_face", False):
            events.append("HAND_FACE")

        # Backward compatibility
        elif behaviour.get("hand_face", False):
            events.append("HAND_FACE")

        # ==========================================
        # 7. LEANING / SHOULDER TILT
        # ==========================================

        shoulder_tilt = behaviour.get(
            "shoulder_tilt",
            0.0
        )

        if shoulder_tilt > 25:
            events.append("LEANING")

        # Backward compatibility
        elif behaviour.get("leaning", False):
            events.append("LEANING")

        # ==========================================
        # 8. STUDENT ABSENCE
        # ==========================================

        if behaviour.get("is_absent", False):
            events.append("STUDENT_ABSENT")

        # ==========================================
        # 9. NORMAL BEHAVIOUR
        # ==========================================

        if len(events) == 0:
            events.append("NORMAL")

        # ==========================================
        # UPDATE RISK ENGINE
        # ==========================================

        for event in events:

            self.risk_engine.update(
                student_id,
                event
            )

        return events