import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import io

# --- 1. CONFIGURATION AND UTILITIES ---

# Define the class names based on the user's custom model training
# We use the user's provided class map: {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
CLASS_NAMES = {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}

# Use the user's custom trained model path.
MODEL_PATH = "yolov8m-football_ball_only.pt"

# --- 2. CORE PROCESSING LOGIC ---

@st.cache_resource
def load_model():
    """Loads the YOLO model only once and caches it."""
    try:
        st.info(f"Attempting to load YOLO model: {MODEL_PATH}")
        # The ultralytics library handles downloading common models if not found locally
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model. Check MODEL_PATH or network connection. Error: {e}")
        st.stop()

def process_video(uploaded_video_file, model, team_a_ids):
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_video_file.read())
        video_path = tfile.name

    output_video_path = os.path.join(tempfile.gettempdir(), "football_tracking_output.mp4")
    output_csv_path = os.path.join(tempfile.gettempdir(), "ball_possession_log.csv")

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Use 'mp4v' for H.264 compatibility which is widely supported
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    possession_log = []

    results = model.track(
        source=video_path,
        conf=0.4,
        iou=0.5,
        persist=True,
        tracker="bytetrack.yaml",
        stream=True,
        verbose=False
    )

    frame_num = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0, text="Processing Frames...")

    for frame_data in results:
        frame_num += 1
        frame = frame_data.orig_img.copy()

        # Update progress bar
        if total_frames > 0:
            progress_bar.progress(min(frame_num / total_frames, 1.0))

        if frame_data.boxes.id is None:
            out.write(frame)
            continue

        try:
            boxes = frame_data.boxes.xyxy.cpu().numpy()
            classes = frame_data.boxes.cls.cpu().numpy().astype(int)
            ids = frame_data.boxes.id.cpu().numpy().astype(int)
        except Exception:
            # Handle cases where boxes.id might not be available or tracking fails
            out.write(frame)
            continue

        balls, players = [], []

        for box, cls, track_id in zip(boxes, classes, ids):
            x1, y1, x2, y2 = map(int, box)
            track_id_int = int(track_id)

            # Determine Team Color and Label
            if cls in [1, 2]: # Player or Goalkeeper
                if track_id_int in team_a_ids:
                    color = (255, 100, 0) # Team A: Blue
                    team_label = "Team A"
                else:
                    color = (0, 0, 255) # Team B: Red
                    team_label = "Team B"

                players.append((track_id_int, (x1, y1, x2, y2)))
                # Draw the team bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Draw the team ID
                cv2.putText(frame, f"{team_label} ID {track_id_int}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            elif cls == 0: # Ball
                balls.append((track_id_int, (x1, y1, x2, y2)))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # --- Possession Calculation ---
        ball_owner_id = None
        min_dist = None
        possession_detected = False
        POSSESSION_THRESHOLD = 80 # Heuristic value from original code

        if len(balls) > 0 and len(players) > 0:
            # Get ball center (assuming one ball)
            ball_box = balls[0][1]
            bx1, by1, bx2, by2 = ball_box
            ball_center = np.array([(bx1 + bx2) / 2, (by1 + by2) / 2])

            min_dist = 1e9
            for player_id, player_box in players:
                # Get player center
                px1, py1, px2, py2 = player_box
                player_center = np.array([(px1 + px2) / 2, (py1 + py2) / 2])

                # Calculate distance
                dist = np.linalg.norm(ball_center - player_center)

                if dist < min_dist:
                    min_dist = dist
                    ball_owner_id = player_id

            # Detect and highlight possession
            if min_dist < POSSESSION_THRESHOLD:
                possession_detected = True
                for player_id, player_box in players:
                    if player_id == ball_owner_id:
                        px1, py1, px2, py2 = player_box
                        # Highlight the possessing player in GREEN (Override Team Color)
                        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 4) # Thicker green box
                        # Display "Has the Ball" text
                        cv2.putText(frame, "Has the Ball", (px1, py2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        break

        # Log Data
        owner_team = None
        if ball_owner_id is not None and possession_detected:
             owner_team = "Team A" if ball_owner_id in team_a_ids else "Team B"

        possession_log.append({
            "frame": frame_num,
            "ball_owner_id": int(ball_owner_id) if ball_owner_id is not None and possession_detected else None,
            "owner_team": owner_team,
            "distance_to_ball": float(min_dist) if min_dist is not None else None
        })

        out.write(frame)

    cap.release()
    out.release()
    os.unlink(video_path)

    # Create DataFrame and save CSV
    df = pd.DataFrame(possession_log)
    df.to_csv(output_csv_path, index=False)

    return output_video_path, output_csv_path, df


# --- 3. STREAMLIT APP UI ---
def streamlit_app():
    # Load the model early and cache it
    model = load_model()

    # 1. Page Config and Background (Using Unsplash image of a pitch)
    st.set_page_config(layout="wide")

    st.markdown("""
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1549477540-1e582e3b2075?q=80&w=2670&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
            background-size: cover;
            background-attachment: fixed;
            color: #ffffff;
            font-family: 'Inter', sans-serif;
        }
        .main-title {
            text-align: center;
            font-size: 3em;
            font-weight: 700;
            color: #FFD700; /* Gold */
            text-shadow: 3px 3px 6px #000000;
            padding-top: 20px;
            padding-bottom: 20px;
        }
        /* Style containers for better readability on the background */
        .block-container {
            background-color: rgba(0, 0, 0, 0.6);
            border-radius: 10px;
            padding: 20px !important;
        }
        .stVideo {
            border: 2px solid #FFD700;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown('<div class="main-title">Football Detection & Tracking</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Layout for inputs and preview
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Upload Video & Team Configuration")
        uploaded_file = st.file_uploader("Upload an MP4 Video of a Football Match", type=["mp4"])

        # Team Assignment Input
        st.markdown("---")
        st.markdown("**Team ID Assignment**")
        team_a_ids_input = st.text_input(
            "Enter Player IDs for Team A (comma-separated, e.g., 1, 3, 5)",
            value="1, 2, 5, 8, 11",
            help="These are the tracking IDs assigned dynamically by the tracker. All other tracked players will be Team B."
        )

        # Convert input string to a set of integers
        team_a_ids = set()
        try:
            if team_a_ids_input:
                # Filter out non-digit strings and convert to int
                team_a_ids = set(map(int, [id.strip() for id in team_a_ids_input.split(',') if id.strip().isdigit()]))
        except ValueError:
            st.error("Invalid input for Team A IDs. Please use comma-separated numbers.")
            team_a_ids = set()

        st.markdown(f"Team A IDs configured: **{', '.join(map(str, sorted(list(team_a_ids))))}**")
        st.markdown("---")


    # Pre-Analysis Video Preview
    with col2:
        if uploaded_file is not None:
            st.subheader("2. Original Video Preview")
            st.video(uploaded_file)
            st.success("Video uploaded successfully!")
        else:
            st.info("Please upload a video to enable the analysis.")


    st.markdown("---")

    # Processing Button
    if uploaded_file is not None and len(team_a_ids) > 0:
        if st.button("Start Tracking & Possession Analysis", key="start_analysis", type="primary"):
            try:
                # Execute core logic
                output_video_path, output_csv_path, df_log = process_video(
                    uploaded_file, model, team_a_ids
                )

                st.success("Analysis Complete!")
                st.markdown("---")

                # --- Output Section ---
                st.subheader("3. Processed Video Output")
                output_col1, output_col2 = st.columns([1, 1])

                with output_col1:
                    # Video Display
                    with open(output_video_path, 'rb') as f:
                        output_video_bytes = f.read()
                    st.video(output_video_bytes)

                    # Download button for Video
                    st.download_button(
                        label="Download Processed Video (MP4)",
                        data=output_video_bytes,
                        file_name="football_tracking_output.mp4",
                        mime="video/mp4",
                        type="secondary"
                    )

                with output_col2:
                    st.subheader("4. Ball Possession Log (CSV)")
                    # Display DataFrame
                    st.dataframe(df_log, use_container_width=True)

                    # Download button for CSV
                    csv_file = df_log.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Possession Log (CSV)",
                        data=csv_file,
                        file_name="ball_possession_log.csv",
                        mime="text/csv",
                        type="secondary"
                    )

            except Exception as e:
                st.error("An error occurred during video processing. See console for details.")
                st.exception(e)

    elif uploaded_file is None:
        st.info("Upload a video file to enable the analysis.")
    elif uploaded_file is not None and len(team_a_ids) == 0:
        st.warning("Please ensure valid Player IDs are entered for Team A configuration.")

if __name__ == '__main__':
    streamlit_app()
