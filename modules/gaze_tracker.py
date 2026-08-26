import cv2

class GazeTracker:
    """
    GazeTracker calculates the candidate's 3D head pose and gaze vector line 
    using facial landmark orientations to monitor looking away patterns.
    """
    def __init__(self):
        pass

    def estimate_head_pose(self, face_landmarks: list, crop_w: int, crop_h: int) -> tuple:
        """
        Estimates yaw, pitch, roll angles and projects a 3D gaze vector on the 2D plane.
        """
        pass

    def is_looking_away(self, yaw: float, pitch: float, yaw_threshold: float, pitch_threshold: float) -> bool:
        """
        Checks if calculated gaze angles exceed standard thresholds, indicating lookup/lookaside actions.
        """
        pass
