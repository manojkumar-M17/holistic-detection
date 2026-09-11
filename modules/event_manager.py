"""
Event Manager
--------------
Central coordinator for suspicious activity events.

Responsibilities:
- Log incidents
- Update timeline
- Save evidence
- Send notifications
- Update dashboard

Author: AI Exam Hall Monitoring System
"""

from datetime import datetime


class EventManager:

    def __init__(
        self,
        db_manager=None,
        timeline=None,
        notifier=None,
        logger=None
    ):

        self.db_manager = db_manager
        self.timeline = timeline
        self.notifier = notifier
        self.logger = logger


    # ==========================================
    # GET SEVERITY
    # ==========================================

    def get_severity(self, risk_score):

        if risk_score < 20:
            return "NORMAL"

        elif risk_score < 40:
            return "LOW"

        elif risk_score < 60:
            return "MEDIUM"

        elif risk_score < 80:
            return "HIGH"

        else:
            return "CRITICAL"


    # ==========================================
    # HANDLE EVENT
    # ==========================================

    def handle_event(
        self,
        student_id,
        event,
        risk_score,
        frame=None,
        screenshot_path=None
    ):

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        severity = self.get_severity(
            risk_score
        )

        # Build incident payload
        incident = {

            "student_id": student_id,

            "event": event,

            "risk_score": risk_score,

            "severity": severity,

            "timestamp": timestamp,

            "screenshot": screenshot_path
        }

        print()

        print("=" * 60)

        print("[EVENT DETECTED]")

        print(f"Student ID : {student_id}")

        print(f"Activity   : {event}")

        print(f"Risk Score : {risk_score}")

        print(f"Severity   : {severity}")

        print(f"Time       : {timestamp}")

        print("=" * 60)


        # ======================================
        # UPDATE TIMELINE
        # ======================================

        if self.timeline:

            self.timeline.add_event(
                student_id,
                incident
            )


        # ======================================
        # SAVE TO DATABASE
        # ======================================

        if self.db_manager:

            self.db_manager.insert_incident(
                incident
            )


        # ======================================
        # SEND NOTIFICATION
        # ======================================

        if self.notifier:

            self.notifier.send_alert(
                incident
            )


        # ======================================
        # LOG EVENT
        # ======================================

        if self.logger:

            self.logger.log(
                incident
            )


        return incident