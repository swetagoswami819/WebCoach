import mediapipe as mp
import numpy as np
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PoseEstimator:
    def __init__(self, static_image_mode=False, min_detection_confidence=0.5, model_complexity=1):
        self.pose = mp_pose.Pose(static_image_mode=static_image_mode,
                                 min_detection_confidence=min_detection_confidence,
                                 model_complexity=model_complexity)

    def process(self, image):
        # expects BGR image
        img_rgb = image[:, :, ::-1]
        results = self.pose.process(img_rgb)
        return results

    def compute_angles(self, landmarks, image_shape):
        """
        Compute common joint angles (in degrees). Return dict with keys:
        left/right knee angle, elbow angle, hip angle, back_angle etc.
        landmarks: mediapipe landmark object
        image_shape: (h, w, c)
        """
        h, w = image_shape[0], image_shape[1]
        lm = {i: (int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h))
              for i in range(len(landmarks.landmark))}

        # helper to compute angle between three points (deg)
        def angle(a, b, c):
            # angle at b formed by ba and bc
            ax, ay = a
            bx, by = b
            cx, cy = c
            v1 = (ax - bx, ay - by)
            v2 = (cx - bx, cy - by)
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = math.hypot(v1[0], v1[1])
            mag2 = math.hypot(v2[0], v2[1])
            if mag1*mag2 == 0:
                return 0.0
            cosang = max(-1.0, min(1.0, dot / (mag1*mag2)))
            return math.degrees(math.acos(cosang))

        # Mediapipe landmark indexes
        NOSE = 0
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28

        angles = {}
        try:
            angles["left_knee_angle"] = angle(lm[LEFT_HIP], lm[LEFT_KNEE], lm[LEFT_ANKLE])
            angles["right_knee_angle"] = angle(lm[RIGHT_HIP], lm[RIGHT_KNEE], lm[RIGHT_ANKLE])
            angles["left_elbow_angle"] = angle(lm[LEFT_SHOULDER], lm[LEFT_ELBOW], lm[LEFT_WRIST])
            angles["right_elbow_angle"] = angle(lm[RIGHT_SHOULDER], lm[RIGHT_ELBOW], lm[RIGHT_WRIST])
            # hip angle (shoulder-hip-knee)
            angles["left_hip_angle"] = angle(lm[LEFT_SHOULDER], lm[LEFT_HIP], lm[LEFT_KNEE])
            angles["right_hip_angle"] = angle(lm[RIGHT_SHOULDER], lm[RIGHT_HIP], lm[RIGHT_KNEE])
            # back angle: use shoulder-hip-ankle as proxy for torso/back tilt
            mid_shoulder = ((lm[LEFT_SHOULDER][0]+lm[RIGHT_SHOULDER][0])//2,
                            (lm[LEFT_SHOULDER][1]+lm[RIGHT_SHOULDER][1])//2)
            mid_hip = ((lm[LEFT_HIP][0]+lm[RIGHT_HIP][0])//2,
                       (lm[LEFT_HIP][1]+lm[RIGHT_HIP][1])//2)
            mid_ankle = ((lm[LEFT_ANKLE][0]+lm[RIGHT_ANKLE][0])//2,
                         (lm[LEFT_ANKLE][1]+lm[RIGHT_ANKLE][1])//2)
            angles["back_angle"] = angle(mid_shoulder, mid_hip, mid_ankle)
            # knee vs ankle x positions for knee caving check (normalized)
            angles["left_knee_ankle_dx"] = (lm[LEFT_KNEE][0] - lm[LEFT_ANKLE][0]) / max(1, w)
            angles["right_knee_ankle_dx"] = (lm[RIGHT_KNEE][0] - lm[RIGHT_ANKLE][0]) / max(1, w)
            # center line distances for valgus (approx)
            center_x = w / 2
            angles["left_knee_center_dist"] = abs(lm[LEFT_KNEE][0] - center_x) / w
            angles["right_knee_center_dist"] = abs(lm[RIGHT_KNEE][0] - center_x) / w
        except Exception:
            pass

        return angles
