import streamlit as st
import cv2
import tempfile
import base64
import os
import time
from ultralytics import YOLO

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
    progress = st.progress(0)

    # --------------------- LOAD MODEL ---------------------
    model_path = "yolov8m-football_ball_only.pt"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please ensure 'yolov8m-football_ball_only.pt' is in the same directory.")
        st.stop()

    model = YOLO(model_path)

    # --------------------- PROCESS VIDEO ---------------------
    try:
        for percent in range(0, 50, 10):
            time.sleep(0.2)
            progress.progress(percent)

        results = model.track(
            source=video_path,
            save=True,
            project=tempfile.gettempdir(),
            name="football_tracking",
            tracker="bytetrack.yaml",
            show=False
        )

        for percent in range(50, 101, 10):
            time.sleep(0.2)
            progress.progress(percent)

    except Exception as e:
        st.error(f"❌ Error during tracking: {e}")
        st.stop()

    # --------------------- FIND OUTPUT FILE SAFELY ---------------------
    output_dir = os.path.join(tempfile.gettempdir(), "football_tracking")
    output_path = None

    if os.path.exists(output_dir):
        for root, _, files in os.walk(output_dir):
            for f in files:
                if f.endswith(".mp4"):
                    output_path = os.path.join(root, f)
                    break
    else:
        st.error("❌ Output directory not found.")
        st.stop()

    if output_path is None or not os.path.exists(output_path):
        st.error("❌ No output video was generated. Please check if detections were made.")
        st.stop()

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
