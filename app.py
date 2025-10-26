import sys
import os

# Add the absolute path to the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))




import streamlit as st
import numpy as np
import pyttsx3
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import cv2
import av  # you need this too for VideoFrame handling

# fixed imports
from vision.pose_estimator import PoseEstimator
from utils.corrections import check_form
from utils.rep_counter import RepCounter
from utils.utils import draw_landmarks, draw_text


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="WebcamCoach — real-time form correction", layout="wide")
st.title("WebcamCoach — Real-time Form Correction & Rep Counter (Webcam only)")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Settings")
    exercise = st.selectbox("Exercise", ["Squat", "Push-up", "Deadlift"])
    show_overlay = st.checkbox("Show drawing overlay", value=True)
    feedback_voice = st.checkbox("Enable voice feedback", value=False)
    sensitivity = st.slider("Sensitivity (how strict corrections are)", 0.5, 1.5, 1.0, 0.1)
    st.markdown("""
    **Notes**
    - Place your webcam so your whole body is visible.
    - Good lighting improves pose estimation accuracy.
    """)

with col2:
    st.header("Session")
    st.markdown("Reps counted and form hints will appear here.")
    rep_placeholder = st.empty()
    hints_placeholder = st.empty()
    # runtime stats
    stats = {"reps": 0, "last_hint": ""}

# Setup TTS engine
tts_engine = None
if feedback_voice:
    try:
        tts_engine = pyttsx3.init()
    except Exception as e:
        st.warning(f"Voice feedback could not be initialized: {e}")
        tts_engine = None

# Pose estimator & rep counter (shared across frames)
pose_est = PoseEstimator(static_image_mode=False, min_detection_confidence=0.6, model_complexity=1)
rep_counter = RepCounter(exercise=exercise, down_angle=70, up_angle=160)

# Update rep_counter if exercise changes (reinitialize)
def reset_session_for_exercise(ex):
    global rep_counter, stats
    # default angle thresholds per exercise (tweakable)
    if ex == "Squat":
        rep_counter = RepCounter(exercise=ex, down_angle=80, up_angle=160)
    elif ex == "Push-up":
        rep_counter = RepCounter(exercise=ex, down_angle=50, up_angle=160)
    elif ex == "Deadlift":
        rep_counter = RepCounter(exercise=ex, down_angle=40, up_angle=160)
    stats = {"reps": 0, "last_hint": ""}

reset_session_for_exercise(exercise)

# If the user changes exercise in the UI, reset counters
if "last_exercise" not in st.session_state:
    st.session_state["last_exercise"] = exercise
if st.session_state["last_exercise"] != exercise:
    reset_session_for_exercise(exercise)
    st.session_state["last_exercise"] = exercise

# -----------------------
# Video Transformer
# -----------------------
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.pose = pose_est
        self.rep_counter = rep_counter
        self.sensitivity = sensitivity
        self.show_overlay = show_overlay
        self.exercise = exercise
        self.tts = tts_engine

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # flip horizontally for mirror effect
        img = cv2.flip(img, 1)

        results = self.pose.process(img)

        # compute landmark-based angles dictionary for checks
        angles = {}
        if results.pose_landmarks:
            angles = self.pose.compute_angles(results.pose_landmarks, img.shape)

            # rep counting using one angle (configure per exercise)
            primary_angle = None
            if self.exercise == "Squat":
                # knee angle (right and left avg) or hip-knee-ankle
                primary_angle = (angles.get("right_knee_angle") + angles.get("left_knee_angle")) / 2.0
            elif self.exercise == "Push-up":
                primary_angle = (angles.get("right_elbow_angle") + angles.get("left_elbow_angle")) / 2.0
            elif self.exercise == "Deadlift":
                primary_angle = (angles.get("back_angle"))  # custom back/hip angle

            rep_event = self.rep_counter.update(primary_angle)
            if rep_event == "up_to_down":
                # optional mid-rep feedback
                pass
            elif rep_event == "down_to_up":
                stats["reps"] += 1
                rep_placeholder.markdown(f"**Reps:** {stats['reps']}")
                # voice
                if self.tts:
                    try:
                        self.tts.say("Good rep")
                        self.tts.runAndWait()
                    except Exception:
                        pass

            # form corrections
            hints = check_form(self.exercise, angles, sensitivity=self.sensitivity)
            if hints:
                stats["last_hint"] = hints[0]  # show first hint prominently
                # voice feedback for serious errors
                if self.tts:
                    try:
                        self.tts.say(hints[0])
                        self.tts.runAndWait()
                    except Exception:
                        pass
            else:
                stats["last_hint"] = "Good form"

            # draw overlay
            if self.show_overlay:
                draw_landmarks(img, results.pose_landmarks, angles)
                draw_text(img, f"Reps: {stats['reps']}", (10, 30))

        else:
            draw_text(img, "No person detected", (10, 30))

        # update UI placeholders (thread-safe via streamlit's script)
        try:
            rep_placeholder.markdown(f"**Reps:** {stats['reps']}")
            hints_placeholder.info(stats["last_hint"])
        except Exception:
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# webrtc streamer
webrtc_ctx = webrtc_streamer(
    key="webcam-coach",
    mode=WebRtcMode.RECVONLY,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown("---")
st.write("Tip: If video doesn't appear, allow camera access and refresh the page.")

