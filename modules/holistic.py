import cv2
import mediapipe.python.solutions.holistic as mp_holistic
import mediapipe.python.solutions.drawing_utils as mp_draw
import time

class HolisticDetector:
    def __init__(self):
        """
        Manages MediaPipe Holistic instances.
        To maintain temporal tracking coherence, we dynamically create and maintain
        separate MediaPipe Holistic instances for each tracked student ID.
        """
        self.mp_holistic = mp_holistic
        self.mp_draw = mp_draw
        self.instances = {}          # student_id -> mp_holistic.Holistic instance
        self.last_active = {}        # student_id -> timestamp of last frame processed
        self.timeout = 10.0          # seconds to keep inactive instances in memory

    def _get_or_create_instance(self, student_id):
        """
        Returns an existing Holistic instance for the given student_id, or creates a new one.
        """
        current_time = time.time()
        self.last_active[student_id] = current_time
        
        # Clean up stale instances to prevent memory leak
        stale_ids = [sid for sid, ltime in self.last_active.items() if current_time - ltime > self.timeout]
        for sid in stale_ids:
            if sid in self.instances:
                self.instances[sid].close()
                del self.instances[sid]
            del self.last_active[sid]
            print(f"[HOLISTIC] Cleaned up inactive MediaPipe instance for student {sid}")

        if student_id not in self.instances:
            print(f"[HOLISTIC] Initializing new MediaPipe Holistic instance for student {student_id}...")
            # static_image_mode=False enables video tracking context for better performance
            self.instances[student_id] = self.mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
        return self.instances[student_id]

    def process_student(self, frame, bbox, student_id):
        """
        Crops the student bounding box, converts it to RGB, runs MediaPipe Holistic,
        and returns the raw landmarks plus local/global coordinates.
        """
        h, w, _ = frame.shape
        x1, y1, x2, y2 = bbox
        
        # Clip bounding box coordinates to frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # If crop is too small or invalid, return None
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return None

        # 1. Extract crop
        crop = frame[y1:y2, x1:x2]
        
        # 2. Preprocess crop: convert to RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # 3. Get MediaPipe instance and process
        holistic_inst = self._get_or_create_instance(student_id)
        results = holistic_inst.process(crop_rgb)
        
        # Assemble result dictionary
        landmark_data = {
            "crop_dims": (x1, y1, x2 - x1, y2 - y1), # (offset_x, offset_y, crop_w, crop_h)
            "face_landmarks": results.face_landmarks,
            "pose_landmarks": results.pose_landmarks,
            "left_hand_landmarks": results.left_hand_landmarks,
            "right_hand_landmarks": results.right_hand_landmarks
        }
        
        return landmark_data

    def release_all(self):
        """
        Closes all active MediaPipe instances.
        """
        for sid, inst in self.instances.items():
            inst.close()
        self.instances.clear()
        self.last_active.clear()
        print("[HOLISTIC] All MediaPipe Holistic resources released.")
