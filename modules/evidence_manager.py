"""
Evidence Manager
----------------
Handles screenshots and video evidence for suspicious activities.

Author : AI Exam Hall Monitoring System
"""

import cv2
import os
from datetime import datetime


class EvidenceManager:

    def __init__(
        self,
        screenshot_dir="screenshots",
        video_dir="videos"
    ):

        self.screenshot_dir = screenshot_dir
        self.video_dir = video_dir

        os.makedirs(self.screenshot_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

    def save_screenshot(self, frame, student_id, event):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"Student_{student_id}_{event}_{timestamp}.jpg"

        path = os.path.join(
            self.screenshot_dir,
            filename
        )

        cv2.imwrite(path, frame)

        return path

    def start_video_recording(
        self,
        frame,
        student_id,
        event,
        fps=20
    ):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"Student_{student_id}_{event}_{timestamp}.mp4"

        path = os.path.join(
            self.video_dir,
            filename
        )

        h, w = frame.shape[:2]

        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )

        return writer, path

    def stop_video_recording(self, writer):

        if writer is not None:
            writer.release()