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

    # --------------------- VIDEO SETTINGS ---------------------
    cap = cv2.VideoCapture(temp_input.name)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # حفظ الفيديو الناتج
    output_path = os.path.join(tempfile.gettempdir(), "processed_football_video.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # --------------------- COLORS ---------------------
    color_ball = (0, 255, 255)
    color_referee = (200, 200, 200)
    color_team_a = (0, 0, 255)
    color_team_b = (255, 0, 0)
    team_colors = {}

    def get_average_color(frame, box):
        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return np.array([0, 0, 0])
        return np.mean(roi.reshape(-1, 3), axis=0)

    def assign_team(player_id, avg_color):
        if player_id not in team_colors:
            if np.mean(avg_color) < 128:
                team_colors[player_id] = "A"
            else:
                team_colors[player_id] = "B"
        return team_colors[player_id]

    # --------------------- PROCESSING ---------------------
    st.info("🚀 Processing video... please wait.")
    progress = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0

    results = model.track(
        source=temp_input.name,
        conf=0.4,
        iou=0.5,
        tracker="botsort.yaml",
        persist=True,
        stream=True
    )

    for frame_data in results:
        frame = frame_data.orig_img.copy()
        frame_num += 1
        progress.progress(frame_num / total_frames)

        if frame_data.boxes.id is None:
            out.write(frame)
            continue

        boxes = frame_data.boxes.xyxy.cpu().numpy()
        classes = frame_data.boxes.cls.cpu().numpy().astype(int)
        ids = frame_data.boxes.id.cpu().numpy().astype(int)

        for box, cls, track_id in zip(boxes, classes, ids):
            x1, y1, x2, y2 = map(int, box)

            # Ball
            if cls == 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_ball, 3)
                cv2.putText(frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, color_ball, 2)

            # Player
            elif cls in [1, 2]:
                avg_color = get_average_color(frame, (x1, y1, x2, y2))
                team = assign_team(track_id, avg_color)
                color = color_team_a if team == "A" else color_team_b
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"Team {team} #{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

            # Referee
            elif cls == 3:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_referee, 2)
                cv2.putText(frame, "Referee", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, color_referee, 2)

        out.write(frame)

    cap.release()
    out.release()

    st.success("✅ Tracking Complete!")

    # --------------------- DISPLAY OUTPUT ---------------------
    st.markdown("<h3 style='text-align:center;'>🎥 Processed Video</h3>", unsafe_allow_html=True)

    # عرض الفيديو في Streamlit
    with open(output_path, "rb") as video_file:
        video_bytes = video_file.read()
        st.video(video_bytes)

    # --------------------- DOWNLOAD BUTTON ---------------------
    b64 = base64.b64encode(video_bytes).decode()
    href = f'<a href="data:video/mp4;base64,{b64}" download="processed_football_video.mp4">⬇️ Download Processed Video</a>'
    st.markdown(href, unsafe_allow_html=True)

else:
    st.info("📥 Please upload a football match video to start tracking.")
