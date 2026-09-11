import cv2
import numpy as np
from collections import deque
import time


class GazeTracker:
    """
    Estimates head pose and stabilizes gaze events.

    Features:
    - Head pose estimation using solvePnP
    - Moving average smoothing
    - Pose confidence validation
    - Minimum duration threshold
    - Event cooldown
    """

    def __init__(
        self,
        smoothing_window=5,
        min_duration=1.0,
        cooldown=2.0,
        min_confidence=0.5
    ):
        # Moving average history
        self.yaw_history = deque(maxlen=smoothing_window)
        self.pitch_history = deque(maxlen=smoothing_window)
        self.roll_history = deque(maxlen=smoothing_window)

        # Event stability
        self.current_direction = "FORWARD"
        self.direction_start_time = None

        # Cooldown tracking
        self.last_event_time = {}

        # Configuration
        self.min_duration = min_duration
        self.cooldown = cooldown
        self.min_confidence = min_confidence

    def estimate_head_pose(self, face_landmarks, crop_w, crop_h):
        """
        Estimate yaw, pitch and roll using facial landmarks.

        Returns:
            yaw, pitch, roll, confidence
        """

        if face_landmarks is None:
            return 0.0, 0.0, 0.0, 0.0

        if len(face_landmarks) < 6:
            return 0.0, 0.0, 0.0, 0.0

        try:
            # Important facial landmark indices
            landmark_ids = [1, 33, 263, 61, 291, 199]

            image_points = []

            for idx in landmark_ids:
                landmark = face_landmarks[idx]

                x = landmark.x * crop_w
                y = landmark.y * crop_h

                image_points.append((x, y))

            image_points = np.array(
                image_points,
                dtype=np.float64
            )

            # Approximate 3D face model points
            model_points = np.array([
                (0.0, 0.0, 0.0),          # Nose
                (-30.0, -30.0, -30.0),    # Left eye
                (30.0, -30.0, -30.0),     # Right eye
                (-25.0, 25.0, -20.0),     # Left mouth
                (25.0, 25.0, -20.0),      # Right mouth
                (0.0, 60.0, -10.0)        # Chin
            ], dtype=np.float64)

            focal_length = crop_w

            camera_matrix = np.array([
                [focal_length, 0, crop_w / 2],
                [0, focal_length, crop_h / 2],
                [0, 0, 1]
            ], dtype=np.float64)

            dist_coeffs = np.zeros((4, 1))

            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return 0.0, 0.0, 0.0, 0.0

            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

            # Convert rotation matrix to angles
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(
                rotation_matrix
            )

            pitch = float(angles[0])
            yaw = float(angles[1])
            roll = float(angles[2])

            # Simple confidence based on valid solvePnP result
            confidence = 1.0

            # -------------------------------
            # MOVING AVERAGE SMOOTHING
            # -------------------------------

            self.yaw_history.append(yaw)
            self.pitch_history.append(pitch)
            self.roll_history.append(roll)

            smooth_yaw = float(np.mean(self.yaw_history))
            smooth_pitch = float(np.mean(self.pitch_history))
            smooth_roll = float(np.mean(self.roll_history))

            return (
                smooth_yaw,
                smooth_pitch,
                smooth_roll,
                confidence
            )

        except Exception as e:
            print(f"[GAZE ERROR] {e}")

            return 0.0, 0.0, 0.0, 0.0

    def get_gaze_direction(
        self,
        yaw,
        pitch,
        yaw_threshold=20,
        pitch_threshold=15
    ):
        """
        Converts head pose angles into a gaze direction.
        """

        if yaw > yaw_threshold:
            return "LOOKING_RIGHT"

        elif yaw < -yaw_threshold:
            return "LOOKING_LEFT"

        elif pitch > pitch_threshold:
            return "LOOKING_DOWN"

        elif pitch < -pitch_threshold:
            return "LOOKING_UP"

        return "FORWARD"

    def is_looking_away(
        self,
        yaw,
        pitch,
        yaw_threshold=20,
        pitch_threshold=15
    ):
        """
        Checks whether the person is looking away.
        """

        return (
            abs(yaw) > yaw_threshold
            or abs(pitch) > pitch_threshold
        )

    def get_stable_gaze_event(
        self,
        yaw,
        pitch,
        confidence=1.0,
        yaw_threshold=20,
        pitch_threshold=15
    ):
        """
        Applies:
        - Pose confidence
        - Minimum duration
        - Cooldown

        Returns:
            Stable gaze event or None
        """

        # -------------------------------
        # POSE CONFIDENCE CHECK
        # -------------------------------

        if confidence < self.min_confidence:
            return None

        direction = self.get_gaze_direction(
            yaw,
            pitch,
            yaw_threshold,
            pitch_threshold
        )

        current_time = time.time()

        # -------------------------------
        # MINIMUM DURATION THRESHOLD
        # -------------------------------

        if direction != self.current_direction:

            self.current_direction = direction
            self.direction_start_time = current_time

            return None

        if self.direction_start_time is None:
            self.direction_start_time = current_time
            return None

        duration = current_time - self.direction_start_time

        if duration < self.min_duration:
            return None

        # Normal forward direction is not suspicious
        if direction == "FORWARD":
            return None

        # -------------------------------
        # COOLDOWN
        # -------------------------------

        last_time = self.last_event_time.get(direction, 0)

        if current_time - last_time < self.cooldown:
            return None

        # store last event time
        self.last_event_time[direction] = current_time

        return direction