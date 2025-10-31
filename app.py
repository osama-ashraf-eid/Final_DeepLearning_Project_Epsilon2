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

# Extra required keys for new ultralytics versions
with_reid: False
model: "osnet_x0_25"
proximity_thresh: 0.5
appearance_thresh: 0.25
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
        global first_team_color, second_team_color
        if first_team_color is None:
            first_team_color = avg_color
            return "A"
        if second_team_color is None:
            dist = np.linalg.norm(avg_color - first_team_color)
            if dist > 40:
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
        min_dist = float('inf')
        nearest_id = None
        for pid, (x1, y1, x2, y2) in player_boxes.items():
            player_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            dist = np.linalg.norm(ball_center - player_center)
            if dist < min_dist:
                min_dist = dist
                nearest_id = pid
        return nearest_id

    # --------------------- PROCESSING ---------------------
    st.info("🚀 Processing video... please wait.")
    progress = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = 0

    results = model.track(
        source=temp_input.name,
        conf=0.4,
        iou=0.5,
        tracker=botsort_path,
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

        player_boxes = {}
        ball_box = None

        for box, cls, track_id in zip(boxes, classes, ids):
            x1, y1, x2, y2 = map(int, box)

            # حافظ على نفس ID لو تغير
            if track_id not in stable_ids:
                stable_ids[track_id] = len(stable_ids) + 1
            display_id = stable_ids[track_id]

            avg_color = get_dominant_color(frame, (x1, y1, x2, y2))

            if cls == 0:  # Ball
                ball_box = (x1, y1, x2, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_ball, 3)
                cv2.putText(frame, "Ball", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, color_ball, 2)

            elif cls in [1, 2, 3]:  # Player / Referee / Goalkeeper
                if is_referee(avg_color):
                    role = "Referee"
                    color = color_referee
                else:
                    team = classify_team(avg_color)
                    role = f"Team {team}"
                    color = color_team_a if team == "A" else color_team_b
                    player_boxes[display_id] = (x1, y1, x2, y2)

                # رسم الصندوق و الـ ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID {display_id} | {role}", (x1, y1 - 15),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

        if ball_box and len(player_boxes) > 0:
            player_with_ball_id = find_player_with_ball(ball_box, player_boxes)
            if player_with_ball_id in player_boxes:
                x1, y1, x2, y2 = player_boxes[player_with_ball_id]
                cv2.putText(frame, "has ball", (x1, y1 - 35),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, color_text_black, 2)

        out.write(frame)

    cap.release()
    out.release()
    st.success("✅ Tracking Complete!")

    # --------------------- DISPLAY OUTPUT ---------------------
    st.markdown("<h3 style='text-align:center;'>🎥 Processed Video</h3>", unsafe_allow_html=True)
    with open(output_path, "rb") as video_file:
        video_bytes = video_file.read()
        st.video(video_bytes)

    b64 = base64.b64encode(video_bytes).decode()
    href = f'<a href="data:video/mp4;base64,{b64}" download="processed_football_video.mp4">⬇️ Download Processed Video</a>'
    st.markdown(href, unsafe_allow_html=True)

else:
    st.info("📥 Please upload a football match video to start tracking.")
