"""
Visualization Module
--------------------
Draws overlays for AI Exam Monitoring System.
"""

import cv2
import mediapipe as mp


class Visualization:

    def __init__(self):
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic


    # ==========================================
    # DRAW STUDENT BOUNDING BOX
    # ==========================================

    def draw_student_box(
        self,
        frame,
        bbox,
        student_id,
        risk,
        level
    ):

        x1, y1, x2, y2 = bbox

        # Ensure integer coordinates
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        color = self.get_color(level)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # Student ID
        cv2.putText(
            frame,
            f"ID: {student_id}",
            (x1, max(20, y1 - 35)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )

        # Risk score
        cv2.putText(
            frame,
            f"Risk: {risk:.1f}%",
            (x1, max(20, y1 - 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )


    # ==========================================
    # DRAW EVENT
    # ==========================================

    def draw_event(
        self,
        frame,
        bbox,
        event
    ):

        x1, y1, x2, y2 = bbox

        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        # Correct OpenCV argument order:
        # image, text, position, font, scale, color, thickness
        cv2.putText(
            frame,
            str(event),
            (x1, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )


    # ==========================================
    # DRAW FPS
    # ==========================================

    def draw_fps(
        self,
        frame,
        fps
    ):

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )


    # ==========================================
    # DRAW HOLISTIC LANDMARKS
    # ==========================================

    def draw_holistic(
        self,
        frame,
        results
    ):

        if results is None:
            return

        if getattr(results, "pose_landmarks", None):

            self.mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                self.mp_styles.get_default_pose_landmarks_style()
            )


        if getattr(results, "left_hand_landmarks", None):

            self.mp_draw.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS
            )


        if getattr(results, "right_hand_landmarks", None):

            self.mp_draw.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS
            )


        if getattr(results, "face_landmarks", None):

            self.mp_draw.draw_landmarks(
                frame,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS
            )


    # ==========================================
    # GET COLOR BASED ON RISK LEVEL
    # ==========================================

    def get_color(
        self,
        level
    ):

        colors = {
            "NORMAL": (0, 255, 0),
            "LOW": (0, 255, 255),
            "MEDIUM": (0, 165, 255),
            "HIGH": (0, 0, 255),
            "CRITICAL": (128, 0, 255)
        }

        if level is None:
            return (255, 255, 255)

        return colors.get(
            str(level).upper(),
            (255, 255, 255)
        )