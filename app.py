import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import os

# --------------------- PAGE SETUP ---------------------
st.set_page_config(page_title="Football Tracking & Analysis", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; color: #00BFFF;'>⚽ Football Match Tracking & Analysis</h1>
    """,
    unsafe_allow_html=True
)

# --------------------- LOCAL IMAGE DISPLAY ---------------------
IMAGE_PATH = "football_img.jpg"
if os.path.exists(IMAGE_PATH):
    st.image(IMAGE_PATH, use_container_width=True, caption="Automated Football Match Analysis using YOLOv8")
else:
    st.warning(f"⚠️ Image file not found: {IMAGE_PATH}")

st.write("---")

# --------------------- MODEL PATH ---------------------
MODEL_PATH = "yolov8m-football_ball_only.pt"
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found: {MODEL_PATH}")
    st.stop()

# Load YOLO model once
model = YOLO(MODEL_PATH)
st.success("✅ Model loaded successfully!")

# --------------------- FILE UPLOAD ---------------------
uploaded_video = st.file_uploader("🎥 Upload a football video (.mp4)", type=["mp4"])

if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        tmp_video.write(uploaded_video.read())
        video_path = tmp_video.name

    st.info("Video uploaded successfully. Starting analysis...")

    # --------------------- OUTPUT VIDEO SETUP ---------------------
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(output_file.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracker = "botsort.yaml"

    results = model.track(
        source=video_path,
        conf=0.4,
        iou=0.5,
        tracker=tracker,
        persist=True,
        stream=True
    )

    # --------------------- ANALYSIS SETUP ---------------------
    color_ball = (0, 255, 255)
    color_referee = (200, 200, 200)
    color_possession = (0, 255, 0)

    last_owner_id = None
    possession_counter = defaultdict(int)
    passes = []
    team_colors = {}
    team_possession_counter = defaultdict(int)
    team_passes_counter = defaultdict(int)

    def get_average_color(frame, box):
        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return np.array([0, 0, 0])
        return np.mean(roi.reshape(-1, 3), axis=0)

    def assign_team(player_id, color):
        if player_id not in team_colors:
            if len(team_colors) == 0:
                team_colors[player_id] = color
            else:
                min_dist = 1e9
                assigned_team = None
                for pid, c in team_colors.items():
                    dist = np.linalg.norm(color - c)
                    if dist < min_dist:
                        min_dist = dist
                        assigned_team = pid
                if min_dist < 40:
                    team_colors[player_id] = team_colors[assigned_team]
                else:
                    team_colors[player_id] = color
        return team_colors[player_id]

    progress_bar = st.progress(0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0

    for frame_data in results:
        frame = frame_data.orig_img.copy()
        processed_frames += 1
        progress_bar.progress(min(processed_frames / frame_count, 1.0))

        if frame_data.boxes.id is None:
            out.write(frame)
            continue

        boxes = frame_data.boxes.xyxy.cpu().numpy()
        classes = frame_data.boxes.cls.cpu().numpy().astype(int)
        ids = frame_data.boxes.id.cpu().numpy().astype(int)

        balls, players = [], []

        for box, cls, track_id in zip(boxes, classes, ids):
            x1, y1, x2, y2 = map(int, box)

            if cls == 0:
                balls.append((track_id, (x1, y1, x2, y2)))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_ball, 2)
                cv2.putText(frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color_ball, 2)

            elif cls in [1, 2]:
                avg_color = get_average_color(frame, (x1, y1, x2, y2))
                team_color = assign_team(track_id, avg_color)
                if np.mean(team_color) < 128:
                    draw_color = (0, 0, 255)
                    team_name = "Team A"
                else:
                    draw_color = (255, 0, 0)
                    team_name = "Team B"
                players.append((track_id, (x1, y1, x2, y2), team_name))
                cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
                cv2.putText(frame, f"{team_name} #{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, draw_color, 2)

            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_referee, 2)
                cv2.putText(frame, "Referee", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, color_referee, 2)

        # Ball possession logic
        current_owner_id = None
        current_owner_team = None
        if len(balls) > 0 and len(players) > 0:
            bx1, by1, bx2, by2 = balls[0][1]
            ball_center = np.array([(bx1 + bx2) / 2, (by1 + by2) / 2])

            min_dist = 1e9
            for player_id, box, team_name in players:
                px1, py1, px2, py2 = box
                player_center = np.array([(px1 + px2) / 2, (py1 + py2) / 2])
                dist = np.linalg.norm(ball_center - player_center)
                if dist < min_dist:
                    min_dist = dist
                    current_owner_id = player_id
                    current_owner_team = team_name

            if min_dist < 90:
                possession_counter[current_owner_id] += 1
                team_possession_counter[current_owner_team] += 1

                if last_owner_id is not None and current_owner_id != last_owner_id:
                    passes.append((last_owner_id, current_owner_id))
                    team_passes_counter[current_owner_team] += 1
                last_owner_id = current_owner_id

        if current_owner_id is not None:
            for player_id, box, team_name in players:
                if player_id == current_owner_id:
                    px1, py1, px2, py2 = box
                    cv2.rectangle(frame, (px1, py1), (px2, py2), color_possession, 4)
                    cv2.putText(frame, f"{team_name} #{player_id} HAS THE BALL",
                                (px1, py1 - 15), cv2.FONT_HERSHEY_COMPLEX, 0.8, color_possession, 3)

        out.write(frame)

    cap.release()
    out.release()

    st.success("✅ Analysis completed successfully!")

    st.video(output_file.name)
    st.download_button("📥 Download Processed Video", data=open(output_file.name, "rb"), file_name="football_analysis.mp4")

else:
    st.warning("Please upload a football video file (.mp4) to start the analysis.")
