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

# --------------------- VIDEO UPLOAD ---------------------
uploaded_video = st.file_uploader("🎬 Upload a football match video", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_input.write(uploaded_video.read())

    # --------------------- MODEL ---------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "yolov8m-football_ball_only.pt")

    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file not found! Expected here: {model_path}")
        st.stop()

    model = YOLO(model_path)

    # --------------------- OUTPUT SETUP ---------------------
    cap = cv2.VideoCapture(temp_input.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_output.name
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # --------------------- COLORS ---------------------
    color_ball = (0, 255, 255)
    color_referee = (0, 255, 0)
    color_goalkeeper = (0, 165, 255)
    color_team_a = (0, 0, 255)
    color_team_b = (255, 0, 0)
    color_text_black = (0, 0, 0)

    first_team_color = None
    second_team_color = None
    stable_ids = {}

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
with_reid: False
model: "osnet_x0_25"
proximity_thresh: 0.5
appearance_thresh: 0.25
        """)

    # --------------------- HELPER FUNCTIONS ---------------------
    def get_safe_box_coords(x1, y1, x2, y2, frame_shape):
        H, W = frame_shape[:2]
        x1c = int(np.clip(x1, 0, W - 1))
        x2c = int(np.clip(x2, 0, W - 1))
        y1c = int(np.clip(y1, 0, H - 1))
        y2c = int(np.clip(y2, 0, H - 1))
        if x2c <= x1c or y2c <= y1c:
            return None
        return x1c, y1c, x2c, y2c

    def get_dominant_color(frame, box, k=2):
        coords = get_safe_box_coords(*box, frame.shape)
        if coords is None:
            return np.array([0, 0, 0], dtype=float)
        x1, y1, x2, y2 = coords
        roi = frame[y1:y2, x1:x2].copy()
        if roi.size == 0:
            return np.array([0, 0, 0], dtype=float)
        roi_pixels = roi.reshape((-1, 3))
        roi_pixels = np.float32(roi_pixels)
        try:
            _, labels, centers = cv2.kmeans(roi_pixels, k, None,
                                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                            10, cv2.KMEANS_RANDOM_CENTERS)
            counts = np.bincount(labels.flatten())
            return centers[np.argmax(counts)]
        except cv2.error:
            return np.mean(roi_pixels, axis=0)

    def classify_team(avg_color):
        global first_team_color, second_team_color
        if first_team_color is None:
            first_team_color = avg_color
            return "A"
        if second_team_color is None:
            if np.linalg.norm(avg_color - first_team_color) > 40:
                second_team_color = avg_color
                return "B"
            else:
                return "A"
        distA = np.linalg.norm(avg_color - first_team_color)
        distB = np.linalg.norm(avg_color - second_team_color)
        return "A" if distA < distB else "B"

    def is_referee(avg_color):
        return (np.mean(avg_color) < 60) or (avg_color[1] > 150 and avg_color[0] < 80)

    def find_player_with_ball(ball_box, player_boxes):
        bx1, by1, bx2, by2 = [int(i) for i in ball_box]
        ball_center = np.array([(bx1 + bx2) / 2, (by1 + by2) / 2])
        min_dist, nearest_id = float('inf'), None
        for pid, (x1, y1, x2, y2) in player_boxes.items():
            player_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            dist = np.linalg.norm(ball_center - player_center)
            if dist < min_dist:
                min_dist, nearest_id = dist, pid
        return nearest_id

    # --------------------- PROCESSING ---------------------
    st.info("🚀 Processing video... please wait.")
    progress = st.progress(0)

    results = model.track(
        source=temp_input.name,
        conf=0.4,
        iou=0.5,
        tracker=botsort_path,
        persist=True,
        stream=True
    )

    frame_idx = 0
    for frame_data in results:
        frame = frame_data.orig_img.copy()
        frame_idx += 1
        progress.progress(frame_idx % 100 / 100)

        if getattr(frame_data, "boxes", None) is None:
            out.write(frame)
            continue

        boxes = frame_data.boxes.xyxy.cpu().numpy()
        classes = frame_data.boxes.cls.cpu().numpy().astype(int)
        ids = frame_data.boxes.id.cpu().numpy().astype(int) if frame_data.boxes.id is not None else range(len(boxes))

        player_boxes = {}
        ball_box = None

        for box, cls, track_id in zip(boxes, classes, ids):
            x1, y1, x2, y2 = map(int, box)
            if track_id not in stable_ids:
                stable_ids[track_id] = len(stable_ids) + 1
            display_id = stable_ids[track_id]

            avg_color = get_dominant_color(frame, (x1, y1, x2, y2))

            if cls == 0:
                ball_box = (x1, y1, x2, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_ball, 3)
                cv2.putText(frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, color_ball, 2)
            else:
                if is_referee(avg_color):
                    role = "Referee"
                    color = color_referee
                else:
                    team = classify_team(avg_color)
                    role = f"Team {team}"
                    color = color_team_a if team == "A" else color_team_b
                    player_boxes[display_id] = (x1, y1, x2, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID {display_id} | {role}", (x1, y1 - 15), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

        if ball_box and len(player_boxes) > 0:
            pid = find_player_with_ball(ball_box, player_boxes)
            if pid in player_boxes:
                x1, y1, x2, y2 = player_boxes[pid]
                cv2.putText(frame, "has ball", (x1, y1 - 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, color_text_black, 2)

        out.write(frame)

    cap.release()
    out.release()
    st.success("✅ Tracking Complete!")

    # --------------------- DISPLAY OUTPUT ---------------------
    with open(output_path, "rb") as f:
        video_bytes = f.read()
        st.video(video_bytes)
        b64 = base64.b64encode(video_bytes).decode()
        href = f'<a href="data:video/mp4;base64,{b64}" download="processed_football_video.mp4">⬇️ Download Processed Video</a>'
        st.markdown(href, unsafe_allow_html=True)

else:
    st.info("📥 Please upload a football match video to start tracking.")
