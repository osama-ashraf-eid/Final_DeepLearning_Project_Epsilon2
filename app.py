import streamlit as st
import os
import tempfile
import cv2
import numpy as np
from collections import defaultdict
import time

# Install this library: pip install ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    st.error("Error: The 'ultralytics' library is not installed. Please install it using: 'pip install ultralytics'")
    st.stop()


# Expected path for the model file. You must place it in the same folder as this file.
MODEL_PATH = "yolov8m-football_ball_only.pt"
TRACKER_CONFIG = "botsort.yaml"

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------

def get_average_color(frame, box):
    """Calculates the average color of the player's Region of Interest (ROI)."""
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return np.array([0, 0, 0])
    h = y2 - y1
    roi_mid = frame[y1 + int(h*0.25):y2 - int(h*0.25), x1:x2]
    if roi_mid.size == 0:
        return np.mean(roi.reshape(-1,3), axis=0)
    return np.mean(roi_mid.reshape(-1, 3), axis=0)


def assign_team(player_id, color, team_colors):
    COLOR_THRESHOLD = 40
    if player_id not in team_colors:
        if not team_colors:
            team_colors[player_id] = color
        else:
            min_dist = 1e9
            assigned_team_id = None
            unique_team_colors = list(set(team_colors.values()))
            for team_color_val in unique_team_colors:
                dist = np.linalg.norm(color - team_color_val)
                if dist < min_dist:
                    min_dist = dist
                    assigned_team_id = team_color_val
            if min_dist < COLOR_THRESHOLD:
                team_colors[player_id] = assigned_team_id
            else:
                team_colors[player_id] = color
    return team_colors[player_id]


@st.cache_resource
def load_yolo_model(path):
    if not os.path.exists(path):
        st.error(f"Error: Model file not found at the specified path: {path}")
        return None
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None


def process_video_tracking(input_video_path, model, output_video_path, status_placeholder):
    status_placeholder.text("Analyzing video...")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        st.error(f"Failed to open video: {input_video_path}")
        return None, {}, {}, {}, {}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    last_owner_id = None
    possession_counter = defaultdict(int)
    passes = []
    team_colors = {}
    team_possession_counter = defaultdict(int)
    team_passes_counter = defaultdict(int)

    color_ball = (0, 255, 255)
    color_referee = (200, 200, 200)
    color_possession = (0, 255, 0)

    results = model.track(
        source=input_video_path,
        conf=0.4,
        iou=0.5,
        tracker=TRACKER_CONFIG,
        persist=True,
        stream=True,
        verbose=False
    )

    team_color_map = {}
    team_name_map = {}

    status_placeholder.text("Processing frames and determining possession...")
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_data in results:
        frame_count += 1
        if frame_count % 50 == 0 and total_frames > 0:
            progress = min(1.0, frame_count / total_frames)
            status_placeholder.progress(progress, text=f"Frame: {frame_count}/{total_frames} (Tracking in progress)")

        frame = frame_data.orig_img.copy()
        if frame_data.boxes.id is None:
            out.write(frame)
            continue

        boxes = frame_data.boxes.xyxy.cpu().numpy()
        classes = frame_data.boxes.cls.cpu().numpy().astype(int)
        ids = frame_data.boxes.id.cpu().numpy().astype(int)

        balls, players = [], []
        for box, cls, track_id in zip(boxes, classes, ids):
            x1, y1, x2, y2 = map(int, box)
            if cls == 0:  # Ball
                balls.append((track_id, (x1, y1, x2, y2)))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_ball, 2)
                cv2.putText(frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color_ball, 2)

            elif cls in [1, 2]:  # Player
                avg_color = get_average_color(frame, (x1, y1, x2, y2))
                assigned_color = assign_team(track_id, avg_color, team_colors)
                team_colors[track_id] = assigned_color

                if not team_color_map:
                    team_color_map[tuple(assigned_color)] = (0, 0, 255)
                    team_name_map[tuple(assigned_color)] = "Team A"
                elif tuple(assigned_color) not in team_color_map:
                    team_color_map[tuple(assigned_color)] = (255, 0, 0)
                    team_name_map[tuple(assigned_color)] = "Team B"

                draw_color = team_color_map.get(tuple(assigned_color), (0, 255, 255))
                team_name = team_name_map.get(tuple(assigned_color), "Unknown Team")
                players.append((track_id, (x1, y1, x2, y2), team_name))
                cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
                cv2.putText(frame, f"{team_name} #{track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, draw_color, 2)

            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_referee, 2)
                cv2.putText(frame, "Referee", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, color_referee, 2)

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

            POSSESSION_DISTANCE_THRESHOLD = 90
            if min_dist < POSSESSION_DISTANCE_THRESHOLD:
                possession_counter[current_owner_id] += 1
                team_possession_counter[current_owner_team] += 1
                if last_owner_id is not None and current_owner_id != last_owner_id:
                    passes.append((last_owner_id, current_owner_id))
                    team_passes_counter[current_owner_team] += 1
                last_owner_id = current_owner_id

            if current_owner_id is not None and min_dist < POSSESSION_DISTANCE_THRESHOLD:
                for player_id, box, team_name in players:
                    if player_id == current_owner_id:
                        px1, py1, px2, py2 = box
                        cv2.rectangle(frame, (px1, py1), (px2, py2), color_possession, 4)
                        cv2.putText(frame, f"{team_name} #{player_id} HAS THE BALL",
                                    (px1, py1 - 15), cv2.FONT_HERSHEY_COMPLEX, 0.8, color_possession, 3)

        out.write(frame)

    cap.release()
    out.release()
    status_placeholder.text("Analysis complete!")
    return output_video_path, possession_counter, team_possession_counter, passes, team_passes_counter


# ----------------------------------------------------------------------
# Streamlit Interface
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="Football Detection & Tracking - YOLOv8",
    layout="wide"
)

# --- Stylish Header and Image ---
st.markdown(
    """
    <div style='text-align:center;'>
        <h1 style='color:#1F77B4; font-size:45px;'>
            ⚽ Football Detection & Tracking
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

IMAGE_PATH = "football_img.jpg"
if os.path.exists(IMAGE_PATH):
    st.markdown(
        f"""
        <div style="
            position: relative;
            width: 100%;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 8px 20px rgba(0,0,0,0.5);
            margin-bottom: 30px;">
            <img src="{IMAGE_PATH}" style="width:100%; filter:brightness(0.7);">
            <div style="
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                color: white;
                font-size: 40px;
                font-weight: bold;
                text-shadow: 2px 2px 10px #000;">
                ⚡ AI-Powered Match Analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning(f"Image file '{IMAGE_PATH}' not found. Please ensure it is in the same directory.")

st.markdown("""
Upload a **football match video** to start the intelligent detection and tracking process.  
You'll get detailed insights about **ball possession, team passes, and overall control** of the game.
""")

# ----------------------------------------------------------------------
# Rest of the app (upload, process, and analysis)
# ----------------------------------------------------------------------

st.warning(f"""
    **Important Note:** The model file (weights file) named **`{MODEL_PATH}`** must be placed 
    in the same folder as this file for the application to work.
""")

model = load_yolo_model(MODEL_PATH)
if model:
    uploaded_file = st.file_uploader("Upload a video file (e.g. .mp4, .avi, .mov)", type=['mp4', 'avi', 'mov'])
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Video")
            with tempfile.NamedTemporaryFile(delete=False) as tfile_in:
                tfile_in.write(uploaded_file.read())
                video_path_in = tfile_in.name
            st.video(video_path_in)

        st.markdown("---")
        if st.button("🚀 Start Analysis and Tracking", type="primary"):
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tfile_out:
                output_path = tfile_out.name

            status_placeholder = st.empty()
            progress_bar = st.progress(0, text="Initializing...")

            with st.spinner("Analysis may take some time..."):
                try:
                    processed_video_path, possession_p, possession_t, passes_list, passes_t = process_video_tracking(
                        video_path_in, model, output_path, status_placeholder
                    )
                    progress_bar.progress(1.0, text="Analysis completed successfully.")
                    with col2:
                        st.subheader("Processed Video (Tracking & Possession)")
                        st.video(processed_video_path)
                    st.markdown("---")
                    st.header("📊 Summary of Possession and Passing Statistics")
                    if possession_t:
                        team_names = list(possession_t.keys())
                        if team_names:
                            total_frames = sum(possession_t.values())
                            col_a, col_b, col_pass = st.columns(3)
                            team_a = team_names[0]
                            col_a.metric(
                                label=f"Possession {team_a}",
                                value=f"{possession_t.get(team_a, 0)} frames",
                                delta=f"{((possession_t.get(team_a, 0)/total_frames)*100):.2f}%"
                            )
                            if len(team_names) > 1:
                                team_b = team_names[1]
                                col_b.metric(
                                    label=f"Possession {team_b}",
                                    value=f"{possession_t.get(team_b, 0)} frames",
                                    delta=f"{((possession_t.get(team_b, 0)/total_frames)*100):.2f}%"
                                )
                            else:
                                col_b.metric("Possession Team B", "N/A")
                            col_pass.metric("Total Passes", f"{len(passes_list)}")
                        st.subheader("Total Passes per Team:")
                        st.dataframe({"Team": list(passes_t.keys()), "Passes Received": list(passes_t.values())}, hide_index=True)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
