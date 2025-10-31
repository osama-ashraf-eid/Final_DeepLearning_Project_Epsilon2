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
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = None

    try:
        for percent in range(0, 50, 10):
            time.sleep(0.2)
            progress.progress(percent)

        # Run YOLO tracking
        results = model.track(
            source=video_path,
            save=True,
            project=output_dir,
            name="football_tracking",
            tracker="bytetrack.yaml",
            show=False
        )

        for percent in range(50, 101, 10):
            time.sleep(0.2)
            progress.progress(percent)

        # Try to find saved video
        output_subdir = os.path.join(output_dir, "football_tracking")
        if os.path.exists(output_subdir):
            for root, _, files in os.walk(output_subdir):
                for f in files:
                    if f.endswith(".mp4"):
                        output_path = os.path.join(root, f)
                        break

    except Exception as e:
        st.error(f"❌ Error during tracking: {e}")
        st.stop()

    # --------------------- FALLBACK: Manual Save ---------------------
    if output_path is None or not os.path.exists(output_path):
        st.warning("⚠️ YOLO did not save output automatically — generating video manually.")
        try:
            results = model.track(source=video_path, tracker="bytetrack.yaml", show=False, save=False)

            manual_path = os.path.join(output_dir, "manual_tracking_output.mp4")
            cap = cv2.VideoCapture(video_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(manual_path, fourcc, fps, (width, height))

            frame_idx = 0
            for result in results:
                if result is not None:
                    frame = result.plot()
                    out.write(frame)
                    frame_idx += 1

            cap.release()
            out.release()
            output_path = manual_path

            st.success(f"✅ Manual video saved successfully ({frame_idx} frames).")

        except Exception as e:
            st.error(f"❌ Manual saving failed: {e}")
            st.stop()

    # --------------------- DISPLAY RESULT ---------------------
    if output_path and os.path.exists(output_path):
        st.success("✅ Tracking Complete!")

        st.markdown("<h3 style='text-align:center;'>🎥 Processed Video</h3>", unsafe_allow_html=True)
        with open(output_path, "rb") as video_file:
            video_bytes = video_file.read()
            st.video(video_bytes)

        # --------------------- DOWNLOAD BUTTON ---------------------
        b64 = base64.b64encode(video_bytes).decode()
        href = f'<a href="data:video/mp4;base64,{b64}" download="processed_football_video.mp4">⬇️ Download Processed Video</a>'
        st.markdown(href, unsafe_allow_html=True)

    else:
        st.error("❌ No output video could be generated even after manual fallback.")

else:
    st.info("📥 Please upload a football match video to start tracking.")
