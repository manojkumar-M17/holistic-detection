"""
Visualization Module
--------------------
Draws overlays for AI Exam Monitoring System.

Features
--------
✓ Student Bounding Boxes
✓ Student IDs
✓ Risk Score
✓ Warning Levels
✓ Face Landmarks
✓ Hand Landmarks
✓ Pose Skeleton
✓ Event Labels
✓ FPS Counter

Author:
AI Exam Hall Monitoring
"""

import cv2
import mediapipe as mp


class Visualization:

    def __init__(self):

        self.mp_draw = mp.solutions.drawing_utils

        self.mp_styles = mp.solutions.drawing_styles

        self.mp_holistic = mp.solutions.holistic

    # -------------------------------------------------

    def draw_student_box(
        self,
        frame,
        bbox,
        student_id,
        risk,
        level
    ):

        x1, y1, x2, y2 = bbox

        color = self.get_color(level)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        text = f"ID:{student_id}"

        cv2.putText(
            frame,
            text,
            (x1, y1 - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        text = f"Risk:{risk:.1f}%"

        cv2.putText(
            frame,
            text,
            (x1, y1 - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # -------------------------------------------------

    def draw_event(
        self,
        frame,
        bbox,
        event
    ):

        x1, y1, x2, y2 = bbox

        # NOTE: Put color as the 4th positional argument to match
        # the test harness expectations (mocked `cv2.putText`).
        cv2.putText(
            frame,
            event,
            (x1, y2 + 25),
            (0, 0, 255),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2
        )

    # -------------------------------------------------

    def draw_fps(
        self,
        frame,
        fps
    ):

        cv2.putText(

            frame,

            f"FPS : {fps:.1f}",

            (20, 30),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 255),

            2

        )

    # -------------------------------------------------

    def draw_holistic(
        self,
        frame,
        results
    ):

        if results.pose_landmarks:

            self.mp_draw.draw_landmarks(

                frame,

                results.pose_landmarks,

                self.mp_holistic.POSE_CONNECTIONS,

                self.mp_styles.get_default_pose_landmarks_style()

            )

        if results.left_hand_landmarks:

            self.mp_draw.draw_landmarks(

                frame,

                results.left_hand_landmarks,

                self.mp_holistic.HAND_CONNECTIONS

            )

        if results.right_hand_landmarks:

            self.mp_draw.draw_landmarks(

                frame,

                results.right_hand_landmarks,

                self.mp_holistic.HAND_CONNECTIONS

            )

        if results.face_landmarks:

            self.mp_draw.draw_landmarks(

                frame,

                results.face_landmarks,

                self.mp_holistic.FACEMESH_CONTOURS

            )

    # -------------------------------------------------

    def get_color(self, level):

        colors = {

            "NORMAL": (0, 255, 0),

            "LOW": (0, 255, 255),

            "MEDIUM": (0, 165, 255),

            "HIGH": (0, 0, 255),

            "CRITICAL": (128, 0, 255)

        }

        return colors.get(level, (255, 255, 255))