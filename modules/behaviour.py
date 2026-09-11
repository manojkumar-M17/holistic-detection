import cv2
import numpy as np
import math


# ==========================================
# 3D GENERIC HEAD MODEL POINTS
# ==========================================

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float32)


# ==========================================
# HEAD POSE ESTIMATION
# ==========================================

def estimate_head_pose(face_landmarks, crop_w, crop_h):
    """
    Estimates head yaw, pitch, and roll.

    Returns:
        yaw, pitch, roll, gaze_line
    """

    if face_landmarks is None:
        return 0.0, 0.0, 0.0, None

    # MediaPipe face landmark indices
    indices = [1, 152, 33, 263, 61, 291]

    image_points = []

    try:
        for idx in indices:
            lm = face_landmarks.landmark[idx]

            x = lm.x * crop_w
            y = lm.y * crop_h

            image_points.append((x, y))

    except (IndexError, AttributeError):
        return 0.0, 0.0, 0.0, None

    image_points = np.array(
        image_points,
        dtype=np.float32
    )

    # Camera parameters
    focal_length = crop_w

    center = (
        crop_w / 2,
        crop_h / 2
    )

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    # Assume no lens distortion
    dist_coeffs = np.zeros(
        (4, 1),
        dtype=np.float32
    )

    # ==========================================
    # SOLVE PNP
    # ==========================================

    try:

        solve_result = cv2.solvePnP(
            MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

    except cv2.error:

        return 0.0, 0.0, 0.0, None

    if not isinstance(solve_result, tuple):

        return 0.0, 0.0, 0.0, None

    if len(solve_result) < 3:

        return 0.0, 0.0, 0.0, None

    success = solve_result[0]
    rotation_vector = solve_result[1]
    translation_vector = solve_result[2]

    if not success:

        return 0.0, 0.0, 0.0, None

    # ==========================================
    # ROTATION MATRIX
    # ==========================================

    rotation_matrix, _ = cv2.Rodrigues(
        rotation_vector
    )

    sy = math.sqrt(
        rotation_matrix[0, 0] ** 2 +
        rotation_matrix[1, 0] ** 2
    )

    singular = sy < 1e-6

    if not singular:

        x = math.atan2(
            rotation_matrix[2, 1],
            rotation_matrix[2, 2]
        )

        y = math.atan2(
            -rotation_matrix[2, 0],
            sy
        )

        z = math.atan2(
            rotation_matrix[1, 0],
            rotation_matrix[0, 0]
        )

    else:

        x = math.atan2(
            -rotation_matrix[1, 2],
            rotation_matrix[1, 1]
        )

        y = math.atan2(
            -rotation_matrix[2, 0],
            sy
        )

        z = 0

    # Convert radians to degrees

    pitch = x * 180.0 / math.pi
    yaw = y * 180.0 / math.pi
    roll = z * 180.0 / math.pi

    # ==========================================
    # GAZE LINE
    # ==========================================

    axis_3d = np.array(
        [(0, 0, 100.0)],
        dtype=np.float32
    )

    nose_tip_2d = image_points[0]

    try:

        projected_points_2d, _ = cv2.projectPoints(
            axis_3d,
            rotation_vector,
            translation_vector,
            camera_matrix,
            dist_coeffs
        )

        target_point_2d = (
            int(projected_points_2d[0][0][0]),
            int(projected_points_2d[0][0][1])
        )

    except cv2.error:

        target_point_2d = None

    gaze_line = None

    if target_point_2d is not None:

        gaze_line = (
            nose_tip_2d,
            target_point_2d
        )

    return yaw, pitch, roll, gaze_line


# ==========================================
# STUDENT BEHAVIOUR ANALYSIS
# ==========================================

def analyze_student_behaviour(
    landmark_data,
    correlated_objects=None
):
    """
    Analyzes student behaviour using:

    - Head pose
    - Gaze direction
    - Stable gaze events
    - Shoulder tilt
    - Standing detection
    - Hand-to-face proximity
    - Student absence
    - Forbidden objects
    """

    # ==========================================
    # DEFAULT FEATURES
    # ==========================================

    features = {

        # Head pose
        "yaw": 0.0,
        "pitch": 0.0,
        "roll": 0.0,

        "gaze_line": None,

        # Gaze tracking
        "gaze_direction": "UNKNOWN",
        "gaze_event": None,
        "gaze_confidence": 0.0,
        "is_looking_away": False,

        # Posture
        "shoulder_tilt": 0.0,
        "is_standing": False,
        "standing_val": 0.0,

        # Hands
        "hands_detected": False,
        "hand_near_face": False,
        "hand_near_face_val": 999.0,

        # Presence
        "is_absent": False,

        # Objects
        "has_forbidden_object": False,
        "forbidden_objects": []
    }

    # ==========================================
    # FORBIDDEN OBJECT DETECTION
    # ==========================================

    if correlated_objects:

        features["has_forbidden_object"] = True

        features["forbidden_objects"] = [

            obj.get("label", "UNKNOWN")

            for obj in correlated_objects

        ]

    # ==========================================
    # NO LANDMARK DATA
    # ==========================================

    if landmark_data is None:

        features["is_absent"] = True

        return features

    # ==========================================
    # CHECK AVAILABLE LANDMARKS
    # ==========================================

    has_face = (
        landmark_data.get(
            "face_landmarks"
        ) is not None
    )

    has_pose = (
        landmark_data.get(
            "pose_landmarks"
        ) is not None
    )

    # Student bounding box exists,
    # but no face or pose detected

    if not has_face and not has_pose:

        features["is_absent"] = True

    # ==========================================
    # CROP DIMENSIONS
    # ==========================================

    crop_dims = landmark_data.get(
        "crop_dims"
    )

    if not crop_dims:

        return features

    _, _, crop_w, crop_h = crop_dims

    # ==========================================
    # 1. GAZE ANALYSIS
    # ==========================================

    gaze_data = landmark_data.get(
        "gaze"
    )

    # ------------------------------------------
    # USE GAZE TRACKER RESULT
    # ------------------------------------------

    if gaze_data:

        features["yaw"] = gaze_data.get(
            "yaw",
            0.0
        )

        features["pitch"] = gaze_data.get(
            "pitch",
            0.0
        )

        features["roll"] = gaze_data.get(
            "roll",
            0.0
        )

        features["gaze_direction"] = gaze_data.get(
            "direction",
            "UNKNOWN"
        )

        features["gaze_event"] = gaze_data.get(
            "event",
            None
        )

        features["gaze_confidence"] = gaze_data.get(
            "confidence",
            0.0
        )

        # Confirmed stable gaze event

        if features["gaze_event"] is not None:

            features["is_looking_away"] = True

    # ------------------------------------------
    # FALLBACK HEAD POSE
    # ------------------------------------------

    elif has_face:

        yaw, pitch, roll, gaze_line = estimate_head_pose(

            landmark_data["face_landmarks"],

            crop_w,

            crop_h

        )

        features["yaw"] = yaw
        features["pitch"] = pitch
        features["roll"] = roll
        features["gaze_line"] = gaze_line

        # Basic looking-away detection

        if abs(yaw) > 20 or abs(pitch) > 15:

            features["is_looking_away"] = True

    # ==========================================
    # 2. SHOULDER TILT + STANDING DETECTION
    # ==========================================

    if has_pose:

        pose = landmark_data[
            "pose_landmarks"
        ]

        try:

            # MediaPipe pose indices
            left_sh = pose.landmark[11]
            right_sh = pose.landmark[12]

            dx = (
                right_sh.x -
                left_sh.x
            ) * crop_w

            dy = (
                right_sh.y -
                left_sh.y
            ) * crop_h

            tilt = math.atan2(
                dy,
                dx
            ) * 180.0 / math.pi

            tilt = abs(tilt)

            if tilt > 90:

                tilt = 180 - tilt

            features["shoulder_tilt"] = tilt

            # Standing estimation

            avg_shoulder_y = (

                left_sh.y +

                right_sh.y

            ) / 2

            features["standing_val"] = (
                avg_shoulder_y
            )

            if avg_shoulder_y < 0.22:

                features["is_standing"] = True

        except (IndexError, AttributeError):

            pass

    # ==========================================
    # 3. HAND-TO-FACE PROXIMITY
    # ==========================================

    left_hand = landmark_data.get(
        "left_hand_landmarks"
    )

    right_hand = landmark_data.get(
        "right_hand_landmarks"
    )

    if left_hand is not None or right_hand is not None:

        features["hands_detected"] = True

        # Default face center

        face_x = crop_w / 2
        face_y = crop_h / 2

        # Use nose position when available

        if has_face:

            try:

                nose = (
                    landmark_data[
                        "face_landmarks"
                    ].landmark[4]
                )

                face_x = nose.x * crop_w

                face_y = nose.y * crop_h

            except (IndexError, AttributeError):

                pass

        min_dist = 999.0

        for hand in [

            left_hand,

            right_hand

        ]:

            if hand is not None:

                try:

                    for lm in hand.landmark:

                        hx = lm.x * crop_w
                        hy = lm.y * crop_h

                        dist = math.hypot(

                            hx - face_x,

                            hy - face_y

                        ) / max(crop_w, 1)

                        if dist < min_dist:

                            min_dist = dist

                except AttributeError:

                    pass

        features["hand_near_face_val"] = (
            min_dist
        )

        if min_dist < 0.35:

            features["hand_near_face"] = True

    # ==========================================
    # RETURN FINAL FEATURES
    # ==========================================

    return features