"""
Event Manager
--------------
Central coordinator for all suspicious activity events.

Responsibilities:
- Log incidents
- Update timeline
- Save evidence
- Send notifications
- Update dashboard

Author : AI Exam Hall Monitoring System
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

    def handle_event(
        self,
        student_id,
        event,
        risk_score,
        frame=None,
        screenshot_path=None
    ):

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        incident = {
            "student_id": student_id,
            "event": event,
            "risk_score": risk_score,
            "timestamp": timestamp,
            "screenshot": screenshot_path
        }

        print("=" * 60)
        print(f"[EVENT]")
        print(f"Student : {student_id}")
        print(f"Activity : {event}")
        print(f"Risk : {risk_score}")
        print("=" * 60)

        if self.timeline:
            self.timeline.add_event(student_id, incident)

        if self.db_manager:
            self.db_manager.insert_incident(incident)

        if self.notifier:
            self.notifier.send_alert(incident)

        if self.logger:
            self.logger.log(incident)

        return incident