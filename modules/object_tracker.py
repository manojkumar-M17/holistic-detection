"""
Object Tracker
--------------
Tracks students using YOLOv8 tracking.

Author : AI Exam Monitoring System
"""

from ultralytics import YOLO


class ObjectTracker:

    def __init__(self, model_path="weights/yolov8n.pt"):

        self.model = YOLO(model_path)

        self.person_class = 0

    def track(self, frame):

        results = self.model.track(

            frame,

            persist=True,

            verbose=False

        )

        students = []

        if not results:
            return students

        result = results[0]

        if result.boxes is None:
            return students

        for box in result.boxes:

            cls = int(box.cls[0])

            if cls != self.person_class:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            track_id = -1

            if box.id is not None:
                track_id = int(box.id[0])

            students.append({

                "student_id": track_id,

                "bbox": (x1, y1, x2, y2),

                "confidence": float(box.conf[0])

            })

        return students