import cv2
import mediapipe.python.solutions.holistic as mp_holistic
import mediapipe.python.solutions.drawing_utils as mp_draw
import time

from modules.gaze_tracker import GazeTracker


class HolisticDetector:
    """
    Manages MediaPipe Holistic instances and integrates
    gaze/head-pose detection for each student.
    """

    def __init__(self):
        """
        Maintains separate MediaPipe Holistic and GazeTracker
        instances for each tracked student.
        """

        self.mp_holistic = mp_holistic
        self.mp_draw = mp_draw

        # student_id -> MediaPipe Holistic instance
        self.instances = {}

        # student_id -> GazeTracker instance
        self.gaze_trackers = {}

        # student_id -> timestamp of last frame processed
        self.last_active = {}

        # Seconds to keep inactive instances
        self.timeout = 10.0

    def _cleanup_stale_instances(self, current_time):
        """
        Removes inactive MediaPipe and GazeTracker instances.
        """

        stale_ids = [
            sid
            for sid, last_time in self.last_active.items()
            if current_time - last_time > self.timeout
        ]

        for sid in stale_ids:

            # Close MediaPipe instance
            if sid in self.instances:
                self.instances[sid].close()
                del self.instances[sid]

            # Remove GazeTracker
            if sid in self.gaze_trackers:
                del self.gaze_trackers[sid]

            # Remove activity record
            if sid in self.last_active:
                del self.last_active[sid]

            print(
                f"[HOLISTIC] Cleaned up inactive "
                f"student {sid}"
            )

    def _get_or_create_instance(self, student_id):
        """
        Returns an existing MediaPipe Holistic instance
        or creates a new one.
        """

        current_time = time.time()

        self.last_active[student_id] = current_time

        self._cleanup_stale_instances(current_time)

        if student_id not in self.instances:

            print(
                f"[HOLISTIC] Initializing MediaPipe "
                f"for student {student_id}"
            )

            self.instances[student_id] = (
                self.mp_holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            )

        return self.instances[student_id]

    def _get_or_create_gaze_tracker(self, student_id):
        """
        Creates a separate GazeTracker for each student.

        This is important because smoothing history,
        duration tracking, and cooldown must be maintained
        independently for each student.
        """

        if student_id not in self.gaze_trackers:

            self.gaze_trackers[student_id] = GazeTracker(
                smoothing_window=5,
                min_duration=1.0,
                cooldown=2.0,
                min_confidence=0.5
            )

            print(
                f"[GAZE] Initialized tracker "
                f"for student {student_id}"
            )

        return self.gaze_trackers[student_id]

    def process_student(self, frame, bbox, student_id):
        """
        Processes one student.

        Steps:
        1. Crop student
        2. Run MediaPipe Holistic
        3. Extract landmarks
        4. Estimate head pose
        5. Detect stable gaze event
        6. Return all data
        """

        h, w, _ = frame.shape

        x1, y1, x2, y2 = bbox

        # ----------------------------------------
        # Clip bounding box
        # ----------------------------------------

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))

        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        # Validate crop
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return None

        # ----------------------------------------
        # Extract student crop
        # ----------------------------------------

        crop = frame[y1:y2, x1:x2]

        crop_h, crop_w = crop.shape[:2]

        # Convert BGR -> RGB
        crop_rgb = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------
        # Run MediaPipe
        # ----------------------------------------

        holistic_inst = self._get_or_create_instance(
            student_id
        )

        results = holistic_inst.process(crop_rgb)

        # ----------------------------------------
        # Default gaze data
        # ----------------------------------------

        gaze_data = {
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0,
            "confidence": 0.0,
            "direction": "UNKNOWN",
            "event": None
        }

        # ----------------------------------------
        # GAZE DETECTION
        # ----------------------------------------

        if results.face_landmarks:

            try:

                gaze_tracker = (
                    self._get_or_create_gaze_tracker(
                        student_id
                    )
                )

                face_landmarks = (
                    results.face_landmarks.landmark
                )

                # Estimate head pose
                (
                    yaw,
                    pitch,
                    roll,
                    confidence
                ) = gaze_tracker.estimate_head_pose(
                    face_landmarks,
                    crop_w,
                    crop_h
                )

                # Get current direction
                direction = (
                    gaze_tracker.get_gaze_direction(
                        yaw,
                        pitch
                    )
                )

                # Get stable event
                event = (
                    gaze_tracker.get_stable_gaze_event(
                        yaw,
                        pitch,
                        confidence
                    )
                )

                gaze_data = {
                    "yaw": round(yaw, 2),
                    "pitch": round(pitch, 2),
                    "roll": round(roll, 2),
                    "confidence": round(confidence, 2),
                    "direction": direction,
                    "event": event
                }

                # Print event only when stable
                if event:

                    print(
                        f"[GAZE EVENT] "
                        f"Student {student_id}: "
                        f"{event}"
                    )

            except Exception as e:

                print(
                    f"[GAZE ERROR] "
                    f"Student {student_id}: {e}"
                )

        # ----------------------------------------
        # Assemble complete result
        # ----------------------------------------

        landmark_data = {

            "crop_dims": (
                x1,
                y1,
                crop_w,
                crop_h
            ),

            "face_landmarks":
                results.face_landmarks,

            "pose_landmarks":
                results.pose_landmarks,

            "left_hand_landmarks":
                results.left_hand_landmarks,

            "right_hand_landmarks":
                results.right_hand_landmarks,

            # New Level 3 data
            "gaze": gaze_data
        }

        return landmark_data

    def release_all(self):
        """
        Releases all MediaPipe resources.
        """

        for sid, inst in self.instances.items():

            try:
                inst.close()

            except Exception as e:

                print(
                    f"[HOLISTIC] Error closing "
                    f"student {sid}: {e}"
                )

        self.instances.clear()
        self.gaze_trackers.clear()
        self.last_active.clear()

        print(
            "[HOLISTIC] All resources released."
        )