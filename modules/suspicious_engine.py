import time
import os
import cv2
import config.config as cfg
from database.db_manager import log_incident

class SuspiciousEngine:
    def __init__(self):
        """
        Maintains tracking states, continuous risk scores (0-100%),
        dynamic threshold overrides, and alert cooldowns per student ID.
        """
        # Dictionary structure: student_id -> {behavior -> start_time}
        self.states = {}
        
        # Real-time risk scores: student_id -> float (0.0 to 100.0)
        self.risk_scores = {}
        
        # Cooldown timer to prevent alert spamming: student_id -> {event_key -> last_logged_time}
        self.log_cooldowns = {}

        # Event state tracking: (student_id, event_type) -> state dict
        # state: {first_seen, last_seen, last_logged, active}
        self.event_states = {}

        # Dynamic sensitivity thresholds (can be updated via dashboard API)
        self.head_yaw_threshold = cfg.HEAD_YAW_THRESHOLD
        self.head_pitch_threshold = cfg.HEAD_PITCH_THRESHOLD
        self.shoulder_tilt_threshold = cfg.SHOULDER_TILT_THRESHOLD
        self.audio_noise_threshold = cfg.AUDIO_NOISE_THRESHOLD
        self.max_allowed_students = cfg.MAX_ALLOWED_STUDENTS
        self.risk_decay_rate = cfg.RISK_DECAY_RATE

    def update_thresholds(self, **kwargs):
        """
        Updates dynamic sensitivity parameters at runtime.
        """
        if "head_yaw_threshold" in kwargs:
            self.head_yaw_threshold = float(kwargs["head_yaw_threshold"])
        if "head_pitch_threshold" in kwargs:
            self.head_pitch_threshold = float(kwargs["head_pitch_threshold"])
        if "shoulder_tilt_threshold" in kwargs:
            self.shoulder_tilt_threshold = float(kwargs["shoulder_tilt_threshold"])
        if "audio_noise_threshold" in kwargs:
            self.audio_noise_threshold = float(kwargs["audio_noise_threshold"])
        if "risk_decay_rate" in kwargs:
            self.risk_decay_rate = float(kwargs["risk_decay_rate"])

    def _get_duration(self, student_id, behavior, active):
        """
        Tracks and returns the duration of a continuous behavior.
        """
        current_time = time.time()
        if student_id not in self.states:
            self.states[student_id] = {}
            
        student_state = self.states[student_id]
        
        if active:
            if behavior not in student_state:
                student_state[behavior] = current_time
                return 0.0
            else:
                return current_time - student_state[behavior]
        else:
            if behavior in student_state:
                del student_state[behavior]
            return 0.0

    def _can_log(self, student_id, behavior):
        """
        Checks if the alert cooldown has passed to prevent duplicate logs.
        """
        current_time = time.time()
        cooldowns = self.log_cooldowns.setdefault(student_id, {})

        # normalize key
        event_key = behavior.upper()

        # get configured cooldown from config if exists
        cd = getattr(__import__("config.config", fromlist=["EVENT_COOLDOWNS"]), "EVENT_COOLDOWNS", {})
        cd_val = cd.get(event_key, cd.get(behavior, 5.0))

        last = cooldowns.get(event_key, 0)
        if current_time - last > cd_val:
            cooldowns[event_key] = current_time
            return True

        return False

    def get_severity(self, risk_score):
        """
        Determines severity string based on risk score percentage.
        """
        if risk_score >= cfg.SEVERITY_THRESHOLDS["CRITICAL"]:
            return "CRITICAL"
        elif risk_score >= cfg.SEVERITY_THRESHOLDS["HIGH"]:
            return "HIGH"
        elif risk_score >= cfg.SEVERITY_THRESHOLDS["MEDIUM"]:
            return "MEDIUM"
        elif risk_score >= cfg.SEVERITY_THRESHOLDS["LOW"]:
            return "LOW"
        return "NORMAL"

    def get_risk_score(self, student_id):
        """
        Returns current risk score for student ID.
        """
        return self.risk_scores.get(student_id, 0.0)

    def check_suspicious(self, student_id, features, frame, bbox, total_student_count=1, audio_metrics=None):
        """
        Applies proctoring rules, updates dynamic risk score (0-100%),
        handles multi-person/proxy rules & audio alerts, and logs incident to DB.
        Returns: (is_suspicious, reason_str, risk_score, severity, category)
        """
        if student_id not in self.risk_scores:
            self.risk_scores[student_id] = 0.0

        current_risk = self.risk_scores[student_id]
        reasons = []
        active_triggers = False
        primary_category = "GENERAL"

        # 0. Proxy / Multiple Persons Alert
        if cfg.ENABLE_PROXY_DETECTION and total_student_count > self.max_allowed_students:
            active_triggers = True
            reasons.append(f"Multiple Persons / Proxy Detected ({total_student_count} present)")
            current_risk = max(current_risk, 90.0)
            primary_category = "PROXY_DETECTION"

        # 0b. Audio Anomaly / Speech Alert
        if audio_metrics and audio_metrics.get("audio_alert"):
            active_triggers = True
            aud_desc = "Speech / Whispering Detected" if audio_metrics.get("is_speech") else "Suspicious Ambient Sound"
            reasons.append(f"Audio Anomaly ({aud_desc})")
            current_risk = min(100.0, current_risk + 6.0)
            if primary_category == "GENERAL":
                primary_category = "AUDIO_ANOMALY"

        if features is None:
            if not active_triggers:
                self.risk_scores[student_id] = max(0.0, current_risk - self.risk_decay_rate)
            return False, "Normal", self.risk_scores[student_id], "NORMAL", "GENERAL"

        # 1. Critical Rule: Forbidden Objects (Cell Phone, Book, Laptop)
        if features.get("has_forbidden_object"):
            active_triggers = True
            obj_names = ", ".join(features["forbidden_objects"])
            reasons.append(f"Unauthorized Item ({obj_names})")
            current_risk = max(current_risk, 95.0) # Instant CRITICAL bump
            primary_category = "PHONE_USAGE" if "Cell Phone" in obj_names else "UNAUTHORIZED_OBJECT"

        # 2. Rule: Student Absence
        if cfg.ENABLE_ABSENCE_DETECTION and features.get("is_absent") and not features.get("has_forbidden_object"):
            absent_duration = self._get_duration(student_id, "absent", True)
            if absent_duration > 2.0:
                active_triggers = True
                reasons.append(f"Student Absent/Left Seat ({absent_duration:.1f}s)")
                current_risk = min(100.0, current_risk + 5.0)
                if primary_category == "GENERAL":
                    primary_category = "STUDENT_ABSENT"
        else:
            self._get_duration(student_id, "absent", False)

        # 3. Rule: Head Turn (Yaw)
        yaw_val = features.get("yaw", 0.0)
        yaw_active = abs(yaw_val) > self.head_yaw_threshold
        yaw_duration = self._get_duration(student_id, "head_turn", yaw_active)
        if yaw_duration > cfg.SUSPICIOUS_DURATION:
            active_triggers = True
            direction = "Right" if yaw_val < 0 else "Left"
            reasons.append(f"Looking {direction} ({yaw_duration:.1f}s)")
            current_risk = min(100.0, current_risk + 3.0)
            if primary_category == "GENERAL":
                primary_category = "HEAD_TURN"

        # 4. Rule: Looking Down / Up (Pitch)
        pitch_val = features.get("pitch", 0.0)
        pitch_active = abs(pitch_val) > self.head_pitch_threshold
        pitch_duration = self._get_duration(student_id, "look_away", pitch_active)
        if pitch_duration > cfg.SUSPICIOUS_DURATION:
            active_triggers = True
            direction = "Down" if pitch_val > 0 else "Up"
            reasons.append(f"Looking {direction} ({pitch_duration:.1f}s)")
            current_risk = min(100.0, current_risk + 3.0)
            if primary_category == "GENERAL":
                primary_category = "LOOKING_DOWN"

        # 5. Rule: Hand Near Face
        hand_active = bool(features.get("hand_near_face", False))
        hand_duration = self._get_duration(student_id, "hand_near_face", hand_active)
        if hand_duration > cfg.SUSPICIOUS_DURATION:
            active_triggers = True
            reasons.append(f"Hand Near Face ({hand_duration:.1f}s)")
            current_risk = min(100.0, current_risk + 2.0)
            if primary_category == "GENERAL":
                primary_category = "HAND_NEAR_FACE"

        # 6. Rule: Body Posture (Leaning / Standing)
        lean_active = float(features.get("shoulder_tilt", 0.0)) > self.shoulder_tilt_threshold
        lean_duration = self._get_duration(student_id, "leaning", lean_active)
        if lean_duration > cfg.SUSPICIOUS_DURATION:
            active_triggers = True
            reasons.append(f"Abnormal Leaning ({lean_duration:.1f}s)")
            current_risk = min(100.0, current_risk + 2.5)
            if primary_category == "GENERAL":
                primary_category = "ABNORMAL_POSTURE"

        if features.get("is_standing", features.get("standing", False)):
            active_triggers = True
            reasons.append("Standing up")
            current_risk = min(100.0, current_risk + 4.0)
            if primary_category == "GENERAL":
                primary_category = "ABNORMAL_POSTURE"

        # Apply decay if no suspicious triggers in current frame
        if not active_triggers:
            current_risk = max(0.0, current_risk - self.risk_decay_rate)
            
        self.risk_scores[student_id] = current_risk
        severity = self.get_severity(current_risk)
        is_suspicious = (current_risk >= cfg.SEVERITY_THRESHOLDS["LOW"])

        if is_suspicious and len(reasons) > 0:
            reason_str = ", ".join(reasons)
            primary_reason = reasons[0].upper()

            # Event state tracking key
            event_key = (student_id, primary_reason)
            now = time.time()

            state = self.event_states.get(event_key)
            if state is None:
                state = {
                    "first_seen": now,
                    "last_seen": now,
                    "last_logged": 0,
                    "active": True
                }
                self.event_states[event_key] = state
            else:
                state["last_seen"] = now
                state["active"] = True

            # Only log if cooldown allows
            if self._can_log(student_id, primary_reason):
                state["last_logged"] = now
                self._trigger_alert(student_id, reason_str, frame, bbox, severity, primary_category, current_risk)

            return True, reason_str, current_risk, severity, primary_category

        return False, "Normal", current_risk, severity, primary_category

    def _trigger_alert(self, student_id, reason, frame, bbox, severity, category, risk_score):
        """
        Saves highlighted screenshot and logs incident with severity & risk score.
        """
        os.makedirs(cfg.SCREENSHOT_DIR, exist_ok=True)
        
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"Student{student_id}_{timestamp_str}.jpg"
        filepath = os.path.join(cfg.SCREENSHOT_DIR, filename)
        
        annotated_screenshot = frame.copy()
        
        color_map = {
            "CRITICAL": (0, 0, 255),
            "HIGH": (0, 69, 255),
            "MEDIUM": (0, 165, 255),
            "LOW": (0, 255, 255)
        }
        alert_color = color_map.get(severity, (0, 0, 255))
        
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated_screenshot, (x1, y1), (x2, y2), alert_color, 3)
            label = f"[{severity}] Student {student_id} (Risk: {risk_score:.0f}%): {reason}"
            cv2.putText(
                annotated_screenshot, label, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, alert_color, 2
            )
        else:
            h, w = annotated_screenshot.shape[:2]
            cv2.rectangle(annotated_screenshot, (10, 10), (w - 10, 60), alert_color, -1)
            label = f"[{severity}] Student {student_id} (Risk: {risk_score:.0f}%): {reason}"
            cv2.putText(
                annotated_screenshot, label, (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
        
        cv2.imwrite(filepath, annotated_screenshot)
        log_incident(student_id, reason, filepath, severity=severity, category=category, risk_score=risk_score)


