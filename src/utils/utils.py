import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def draw_landmarks(image, pose_landmarks, angles=None):
    """
    Draw Mediapipe skeleton and optionally annotate key angles.
    """
    # draw skeleton
    mp_drawing.draw_landmarks(image, pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # annotate some angles if provided
    if angles:
        y = 50
        for k, v in angles.items():
            if isinstance(v, (int, float)):
                cv2.putText(image, f"{k[:12]}:{int(v)}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                y += 15


def draw_text(image, text, org=(10,30), color=(0,0,255)):
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def draw_angle(image, angle_value, pos=(50,50), color=(0,255,0)):
    """Draw an angle value prominently on the image"""
    if angle_value is not None:
        cv2.putText(image, f"{angle_value:.1f}°", pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
