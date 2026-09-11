"""
Evidence Manager
----------------
Handles screenshots and video evidence for suspicious activities.

Features:
- Screenshot saving
- Video recording
- Student-wise evidence folders
- Event cooldown
- Evidence metadata
- Safe file handling

Author: AI Exam Hall Monitoring System
"""

import cv2
import os
import time
from datetime import datetime


class EvidenceManager:

    def __init__(
        self,
        screenshot_dir="screenshots",
        video_dir="videos",
        cooldown_seconds=10
    ):
        """
        Initialize evidence storage directories and cooldown settings.
        """

        self.screenshot_dir = screenshot_dir
        self.video_dir = video_dir
        self.cooldown_seconds = cooldown_seconds

        # Stores the last evidence capture time.
        # Key format: (student_id, event)
        self.last_capture_time = {}

        # Create main directories if they do not exist.
        os.makedirs(
            self.screenshot_dir,
            exist_ok=True
        )

        os.makedirs(
            self.video_dir,
            exist_ok=True
        )

    # ==========================================================
    # CHECK COOLDOWN
    # ==========================================================

    def can_capture(self, student_id, event):
        """
        Check whether evidence can be captured.

        Prevents repeated evidence capture for the same
        student and event during the cooldown period.
        """

        key = (
            student_id,
            str(event).upper()
        )

        current_time = time.time()

        last_time = self.last_capture_time.get(key)

        # No previous capture.
        if last_time is None:
            return True

        elapsed = current_time - last_time

        return elapsed >= self.cooldown_seconds

    # ==========================================================
    # SAVE SCREENSHOT
    # ==========================================================

    def save_screenshot(
        self,
        frame,
        student_id,
        event
    ):
        """
        Save a screenshot for a suspicious activity.

        Returns:
            dict containing evidence metadata,
            or None if saving fails or cooldown is active.
        """

        # Validate frame.
        if frame is None:

            print(
                "[EVIDENCE ERROR] Frame is None."
            )

            return None

        # Check cooldown.
        if not self.can_capture(
            student_id,
            event
        ):

            print(
                f"[EVIDENCE] Cooldown active: "
                f"Student {student_id} - {event}"
            )

            return None

        # Create student-specific folder.
        student_dir = os.path.join(
            self.screenshot_dir,
            f"Student_{student_id}"
        )

        os.makedirs(
            student_dir,
            exist_ok=True
        )

        # Create unique timestamp.
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        # Make event name safe for filenames.
        safe_event = (
            str(event)
            .upper()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

        # Create filename.
        filename = (
            f"{safe_event}_{timestamp}.jpg"
        )

        # Full screenshot path.
        path = os.path.join(
            student_dir,
            filename
        )

        # Save image.
        success = cv2.imwrite(
            path,
            frame
        )

        # Handle write failure.
        if success is False:

            print(
                "[EVIDENCE ERROR] "
                "Failed to save screenshot."
            )

            return None

        # Update cooldown time.
        key = (
            student_id,
            str(event).upper()
        )

        self.last_capture_time[key] = time.time()

        # Create evidence metadata.
        evidence = {

            "student_id": student_id,

            "event": event,

            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "path": path,

            "type": "SCREENSHOT"
        }

        print(
            f"[EVIDENCE SAVED] {path}"
        )

        return evidence

    # ==========================================================
    # START VIDEO RECORDING
    # ==========================================================

    def start_video_recording(
        self,
        frame,
        student_id,
        event,
        fps=20
    ):
        """
        Start video evidence recording.

        Returns:
            tuple: (VideoWriter, video_path)

            If recording fails:
            (None, None)
        """

        # Validate frame.
        if frame is None:

            print(
                "[EVIDENCE ERROR] "
                "Cannot start recording. Frame is None."
            )

            return None, None

        # Create student-specific video folder.
        student_dir = os.path.join(
            self.video_dir,
            f"Student_{student_id}"
        )

        os.makedirs(
            student_dir,
            exist_ok=True
        )

        # Create timestamp.
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        # Safe event name.
        safe_event = (
            str(event)
            .upper()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )

        # Video filename.
        filename = (
            f"{safe_event}_{timestamp}.mp4"
        )

        # Full video path.
        path = os.path.join(
            student_dir,
            filename
        )

        # Get frame dimensions.
        h, w = frame.shape[:2]

        # Video codec.
        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        # Create video writer.
        writer = cv2.VideoWriter(
            path,
            fourcc,
            fps,
            (w, h)
        )

        # Check writer.
        if writer is None:

            print(
                "[EVIDENCE ERROR] "
                "Could not create video writer."
            )

            return None, None

        # Check if the writer opened correctly.
        try:

            if not writer.isOpened():

                print(
                    "[EVIDENCE ERROR] "
                    "Could not open video writer."
                )

                writer.release()

                return None, None

        except AttributeError:
            # Allows compatibility with mocked writers in tests.
            pass

        print(
            f"[VIDEO RECORDING STARTED] {path}"
        )

        return writer, path

    # ==========================================================
    # WRITE VIDEO FRAME
    # ==========================================================

    def write_video_frame(
        self,
        writer,
        frame
    ):
        """
        Write one frame to an active video recording.

        Returns:
            True if frame was written successfully.
            False otherwise.
        """

        if writer is None:

            return False

        if frame is None:

            return False

        try:

            writer.write(frame)

            return True

        except Exception as error:

            print(
                f"[EVIDENCE ERROR] "
                f"Could not write video frame: {error}"
            )

            return False

    # ==========================================================
    # STOP VIDEO RECORDING
    # ==========================================================

    def stop_video_recording(
        self,
        writer
    ):
        """
        Safely stop video recording.
        """

        if writer is not None:

            try:

                writer.release()

                print(
                    "[VIDEO RECORDING STOPPED]"
                )

            except Exception as error:

                print(
                    f"[EVIDENCE ERROR] "
                    f"Could not stop video recording: {error}"
                )

    # ==========================================================
    # GET EVIDENCE METADATA
    # ==========================================================

    def get_evidence_info(
        self,
        student_id,
        event,
        path,
        evidence_type="SCREENSHOT"
    ):
        """
        Create standardized evidence metadata.
        """

        return {

            "student_id": student_id,

            "event": event,

            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "path": path,

            "type": evidence_type
        }

    # ==========================================================
    # CLEAR COOLDOWN
    # ==========================================================

    def clear_cooldown(
        self,
        student_id=None,
        event=None
    ):
        """
        Clear cooldown records.

        Options:
        - No arguments: clear all cooldown records.
        - student_id + event: clear a specific record.
        """

        # Clear all cooldown records.
        if student_id is None and event is None:

            self.last_capture_time.clear()

            print(
                "[EVIDENCE] All cooldown records cleared."
            )

            return

        # Clear a specific cooldown record.
        if student_id is not None and event is not None:

            key = (
                student_id,
                str(event).upper()
            )

            if key in self.last_capture_time:

                del self.last_capture_time[key]

                print(
                    f"[EVIDENCE] Cooldown cleared: "
                    f"Student {student_id} - {event}"
                )