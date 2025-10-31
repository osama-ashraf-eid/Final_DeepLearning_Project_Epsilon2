import streamlit as st
import cv2
import tempfile
import base64
import os
from ultralytics import YOLO
import torch

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
        [data-testid="stHeader"] {background: rgba(0,0,0,0);}
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------- FILE UPLOAD ---------------------
uploaded_file = st.file_uploader("📥 Upload a football match video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    st.info("⚙️ Processing video... Please wait.")

    # --------------------- LOAD MODEL ---------------------
    model_path = "yolov8m-football_ball_only.pt"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please ensure yolov8m-football_ball_only.pt is in the same directory.")
        st.stop()

    model = YOLO(model_path)

    # --------------------- OUTPUT PATH ---------------------
    output_path = os.path.join(tempfile.gettempdir(), "processed_output.mp4")

    # --------------------- PROCESS VIDEO ---------------------
    results = model.track(
        source=video_path,
        save=True,
        project=tempfile.gettempdir(),
        name="football_tracking",
        tracker="bytetrack.yaml",
        show=False
    )

    # Find saved output file
    output_dir = os.path.join(tempfile.gettempdir(), "football_tracking")
    for file in os.listdir(output_dir):
        if file.endswith(".mp4"):
            output_path = os.path.join(output_dir, file)
            break

    # --------------------- SUCCESS MESSAGE ---------------------
    st.success("✅ Tracking Complete!")

    # --------------------- DISPLAY OUTPUT ---------------------
    st.markdown("<h3 style='text-align:center;'>🎥 Processed Video</h3>", unsafe_allow_html=True)
    with open(output_path, "rb") as video_file:
        video_bytes = video_file.read()
        st.video(video_bytes)

    # --------------------- DOWNLOAD BUTTON ---------------------
    b64 = base64.b64encode(video_bytes).decode()
    href = f'<a href="data:video/mp4;base64,{b64}" download="processed_football_video.mp4">⬇️ Download Processed Video</a>'
    st.markdown(href, unsafe_allow_html=True)

else:
    st.info("📥 Please upload a football match video to start tracking.")
