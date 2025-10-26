import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

class VideoProcessor(VideoTransformerBase):
    def recv(self, frame):
        return frame

webrtc_streamer(
    key="test",
    mode="SENDRECV",
    media_stream_constraints={"video": True, "audio": False},
)
