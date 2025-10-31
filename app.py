import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import base64

# --------------------- PAGE SETUP ---------------------
st.set_page_config(page_title="Football Detection & Tracking", layout="wide")

# --------------------- TITLE ---------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #00BFFF; font-size: 48px;'>
        ⚽ Football Detection & Tracking
    </h1>
    """,
    unsafe_allow_html=True
)

# --------------------- BACKGROUND IMAGE ---------------------
st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] {
            background-image: url("https://images.unsplash.com/photo-1518091043644-c1d4457512c6");
            background-size: cover;
            background-position: center;
        }
        h1, h3, p {
            color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------- VIDEO UPLOAD ---------------------
uploaded_video = st.file_uploader("🎬 Upload a football match video", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_input.write(uploaded_video.read())

    # --------------------- MODEL ---------------------
    model_path = "yolov8m-football_ball_only.pt"
    if not os.path.exists(model_path):
        st.error("⚠️ Model file not found! Please upload 'yolov8m-football_ball_only.pt' in the same folder.")
        st.stop()

    model = YOLO(model_path)

    cap = cv2.VideoCapture(temp_input.name)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = os.path.join(tempfile.gettempdir(), "processed_football_video.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # --------------------- COLORS ---------------------
    color_ball = (0, 255, 255)
    color_referee = (0, 255, 0)
    color_goalkeeper = (0, 165, 255)
    color_team_a = (0, 0, 255)
    color_team_b = (255, 0, 0)
    color_text_black = (0, 0, 0)

    first_team_color = None
    second_team_color = None
    stable_ids = {}  # لتثبيت ID كل لاعب

    # --------------------- CUSTOM BoT-SORT CONFIG ---------------------
    botsort_path = os.path.join(tempfile.gettempdir(), "custom_botsort.yaml")
    with open(botsort_path, "w") as f:
        f.write("""
tracker_type: botsort
track_high_thresh: 0.6
track_low_thresh: 0.2
new_track_thresh: 0.7
track_buffer: 120
match_thresh: 0.9
gmc_method: sparseOptFlow
        """)

    # --------------------- HELPER FUNCTIONS ---------------------
    def get_dominant_color(frame, box, k=2):
        x1, y1, x2, y2 = [int(i) for i in box]
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return np.array([0, 0, 0])
        # نختار منتصف الجسم فقط لتجنب العشب
        h_roi, w_roi, _ = roi.shape
        roi = roi[h_roi//3: h_roi*2//3, w_roi//4: w_roi*3//4]
        roi = roi.reshape((-1, 3))
        roi = np.float32(roi)
        _, labels, centers = cv2.kmeans(roi, k, None,
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                        10, cv2.KMEANS_RANDOM_CENTERS)
        counts = np.bincount(labels.flatten())
        dominant = centers[np.argmax(counts)]
        return dominant

    def classify_team(avg_color):
        global first_team_color
