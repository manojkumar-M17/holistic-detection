import cv2
import numpy as np
import math
import mediapipe as mp

# 3D generic head model points for pose estimation
# Coords in mm, matching a generic face model
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float32)

def estimate_head_pose(face_landmarks, crop_w, crop_h):
    """
    Estimates head yaw, pitch, and roll from face landmarks.
    Yaw: turning head left/right (positive is left, negative is right)
    Pitch: looking up/down (positive is down, negative is up)
    Roll: tilting head left/right
    Returns: (yaw, pitch, roll) in degrees, and 2D nose projection lines for drawing.
    """
    if face_landmarks is None:
        return 0.0, 0.0, 0.0, None

    # Landmark indices for pose estimation
    # 1: nose tip, 152: chin, 33: left eye outer corner, 263: right eye outer corner,
    # 61: left mouth corner, 291: right mouth corner
    indices = [1, 152, 33, 263, 61, 291]
    image_points = []
    
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        x = lm.x * crop_w
        y = lm.y * crop_h
        image_points.append((x, y))
        
    image_points = np.array(image_points, dtype=np.float32)

    # Camera internals approximation
    focal_length = crop_w
    center = (crop_w / 2, crop_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion
    
    # Solve PnP (guard against numerical errors / invalid input)
    try:
        solve_result = cv2.solvePnP(
            MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
    except cv2.error:
        return 0.0, 0.0, 0.0, None

    # OpenCV may return different shapes across versions; unpack defensively
    if isinstance(solve_result, tuple) and len(solve_result) >= 3:
        success, rotation_vector, translation_vector = solve_result[0], solve_result[1], solve_result[2]
    else:
        # Unexpected return — fail gracefully
        return 0.0, 0.0, 0.0, None

    if not success:
        return 0.0, 0.0, 0.0, None
        
    # Get Euler angles
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Extract Euler angles from rotation matrix
    sy = math.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
    singular = sy < 1e-6
    
    if not singular:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0
        
    # Convert to degrees
    pitch = x * 180.0 / math.pi
    yaw = y * 180.0 / math.pi
    roll = z * 180.0 / math.pi
    
    # Project a 3D point (e.g. vector pointing out from nose) to 2D screen
    axis_3d = np.array([(0, 0, 100.0)], dtype=np.float32)
    nose_tip_2d = image_points[0]
    
    projected_points_2d, _ = cv2.projectPoints(
        axis_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs
    )
    
    target_point_2d = (int(projected_points_2d[0][0][0]), int(projected_points_2d[0][0][1]))
    
    return yaw, pitch, roll, (nose_tip_2d, target_point_2d)

def analyze_student_behaviour(landmark_data, correlated_objects=None):
    """
    Analyzes holistic landmarks and correlated objects to extract structured behavioral features:
    - Head pose (yaw, pitch, roll)
    - Shoulder tilt angle
    - Hand proximity to face/head
    - Posture state (standing/sitting)
    - Absence check (empty desk / student left seat)
    - Forbidden item detection (Cell phone, Book, Laptop)
    """
    features = {
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,
        "gaze_line": None,
        "shoulder_tilt": 0.0,
        "hands_detected": False,
        "hand_near_face": False,
        "hand_near_face_val": 999.0,
        "is_standing": False,
        "standing_val": 0.0,
        "is_absent": False,
        "has_forbidden_object": False,
        "forbidden_objects": []
    }

    # Process correlated objects (e.g., Cell Phone, Book)
    if correlated_objects:
        features["has_forbidden_object"] = True
        features["forbidden_objects"] = [obj["label"] for obj in correlated_objects]

    if landmark_data is None:
        features["is_absent"] = True
        return features

    # Check for empty landmark crop (Student bounding box present, but MediaPipe found no face/pose)
    has_face = landmark_data.get("face_landmarks") is not None
    has_pose = landmark_data.get("pose_landmarks") is not None
    if not has_face and not has_pose:
        features["is_absent"] = True

    # Get crop dimensions
    _, _, crop_w, crop_h = landmark_data["crop_dims"]
    
    # 1. Head Pose Estimation (Yaw, Pitch, Roll)
    if has_face:
        yaw, pitch, roll, gaze_line = estimate_head_pose(
            landmark_data["face_landmarks"], crop_w, crop_h
        )
        features["yaw"] = yaw
        features["pitch"] = pitch
        features["roll"] = roll
        features["gaze_line"] = gaze_line
        
    # 2. Shoulder Tilt & Standing Check from Pose
    if has_pose:
        pose = landmark_data["pose_landmarks"]
        
        # Left shoulder: 11, Right shoulder: 12
        left_sh = pose.landmark[11]
        right_sh = pose.landmark[12]
        
        # Calculate shoulder tilt angle relative to horizontal plane
        dx = (right_sh.x - left_sh.x) * crop_w
        dy = (right_sh.y - left_sh.y) * crop_h
        tilt = math.atan2(dy, dx) * 180.0 / math.pi
        
        if tilt < 0:
            tilt = -tilt
        if tilt > 90:
            tilt = 180 - tilt
        features["shoulder_tilt"] = tilt
        
        avg_shoulder_y = (left_sh.y + right_sh.y) / 2
        features["standing_val"] = avg_shoulder_y
        
        if avg_shoulder_y < 0.22:
            features["is_standing"] = True

    # 3. Hand-to-Face Proximity
    left_hand = landmark_data["left_hand_landmarks"]
    right_hand = landmark_data["right_hand_landmarks"]
    
    if left_hand is not None or right_hand is not None:
        features["hands_detected"] = True
        
        face_x, face_y = crop_w / 2, crop_h / 2
        if has_face:
            nose = landmark_data["face_landmarks"].landmark[4]
            face_x, face_y = nose.x * crop_w, nose.y * crop_h
            
        min_dist = 999.0
        for hand in [left_hand, right_hand]:
            if hand is not None:
                for lm in hand.landmark:
                    hx, hy = lm.x * crop_w, lm.y * crop_h
                    dist = math.hypot(hx - face_x, hy - face_y) / crop_w
                    if dist < min_dist:
                        min_dist = dist
                        
        features["hand_near_face_val"] = min_dist
        if min_dist < 0.35:
            features["hand_near_face"] = True

    return features

