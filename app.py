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
# Helper Functions (adapted slightly from the provided code)
# ----------------------------------------------------------------------

def get_average_color(frame, box):
    """Calculates the average color of the player's Region of Interest (ROI)."""
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return np.array([0, 0, 0])
    # Using part of the body (upper/center) to avoid shoes or shadows
    h = y2 - y1
    w = x2 - x1
    # We take the middle part of the body (25% to 75% of height)
    roi_mid = frame[y1 + int(h*0.25):y2 - int(h*0.25), x1:x2]
    if roi_mid.size == 0:
        return np.mean(roi.reshape(-1,3), axis=0)

    return np.mean(roi_mid.reshape(-1, 3), axis=0)


def assign_team(player_id, color, team_colors):
    """Assigns the team based on color, improving determination over time."""
    # The threshold value to determine if the color is close to an existing team
    COLOR_THRESHOLD = 40

    if player_id not in team_colors:
        if not team_colors:
            # The first player identified becomes the first team
            team_colors[player_id] = color
        else:
            # Trying to associate the player with the nearest existing team
            min_dist = 1e9
            assigned_team_id = None
            
            # Get the team color from any player belonging to it
            unique_team_colors = list(set(team_colors.values()))

            for team_color_val in unique_team_colors:
                 dist = np.linalg.norm(color - team_color_val)
                 if dist < min_dist:
                     min_dist = dist
                     assigned_team_id = team_color_val

            if min_dist < COLOR_THRESHOLD:
                # If close enough, the same team color is assigned
                team_colors[player_id] = assigned_team_id
            else:
                # Otherwise, a new team color is created (unlikely in football)
                team_colors[player_id] = color

    return team_colors[player_id]


@st.cache_resource
def load_yolo_model(path):
    """Loads the YOLO model and caches it."""
    if not os.path.exists(path):
        st.error(f"Error: Model file not found at the specified path: {path}")
        st.info("Please ensure that the `yolov8m-football_ball_only.pt` file is in the same folder as the application.")
        return None
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None


def process_video_tracking(input_video_path, model, output_video_path, status_placeholder):
    """
    The main function that runs tracking and analysis as defined in the original code. 
    Adapted to work within the Streamlit environment.
    """
    
    status_placeholder.text("Analyzing video...")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        st.error(f"Failed to open video: {input_video_path}")
        return None, {}, {}, {}, {}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Using H.264 (mp4v) to ensure compatibility with web browsers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # Statistical variables
    last_owner_id = None
    possession_counter = defaultdict(int)
    passes = []
    team_colors = {} # Used to store the team color for each player_id
    team_possession_counter = defaultdict(int)
    team_passes_counter = defaultdict(int)

    # Drawing colors (BGR format)
    color_ball = (0, 255, 255) # Cyan
    color_referee = (200, 200, 200) # Gray
    color_possession = (0, 255, 0) # Green

    # Run the tracking process
    results = model.track(
        source=input_video_path,
        conf=0.4,
        iou=0.5,
        tracker=TRACKER_CONFIG,
        persist=True,
        stream=True,
        verbose=False # Do not display tracking results in the console
    )
    
    # Defining fixed colors for the two teams based on the initial average color
    team_color_map = {}
    team_name_map = {}
    
    status_placeholder.text("Processing frames and determining possession...")
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_data in results:
        frame_count += 1
        
        # Update progress bar every 50 frames
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
        
        # Stage One: Tracking and Team Assignment
        for box, cls, track_id in zip(boxes, classes, ids):
            x1, y1, x2, y2 = map(int, box)

            if cls == 0: # Ball
                balls.append((track_id, (x1, y1, x2, y2)))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_ball, 2)
                cv2.putText(frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color_ball, 2)

            elif cls in [1, 2]: # Goalkeeper or Player
                avg_color = get_average_color(frame, (x1, y1, x2, y2))
                assigned_color = assign_team(track_id, avg_color, team_colors)
                
                # Store the fixed team color for this player
                team_colors[track_id] = assigned_color

                # Define fixed drawing colors and team names based on the assigned color
                if not team_color_map:
                    # Assign the first team
                    team_color_map[tuple(assigned_color)] = (0, 0, 255) # Team A: Blue
                    team_name_map[tuple(assigned_color)] = "Team A"
                
                elif tuple(assigned_color) not in team_color_map:
                    # Assign the second team
                    team_color_map[tuple(assigned_color)] = (255, 0, 0) # Team B: Red
                    team_name_map[tuple(assigned_color)] = "Team B"

                draw_color = team_color_map.get(tuple(assigned_color), (0, 255, 255))
                team_name = team_name_map.get(tuple(assigned_color), "Unknown Team")
                
                players.append((track_id, (x1, y1, x2, y2), team_name))
                cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
                cv2.putText(frame, f"{team_name} #{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, draw_color, 2)

            else: # Referee
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_referee, 2)
                cv2.putText(frame, "Referee", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, color_referee, 2)

        # Stage Two: Determining Possession and Calculating Statistics
        current_owner_id = None
        current_owner_team = None
        
        if len(balls) > 0 and len(players) > 0:
            # Assume only one ball (the first detected ball)
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
            
            # Distance threshold for possession determination (can be adjusted)
            POSSESSION_DISTANCE_THRESHOLD = 90
            
            if min_dist < POSSESSION_DISTANCE_THRESHOLD:
                possession_counter[current_owner_id] += 1
                team_possession_counter[current_owner_team] += 1

                if last_owner_id is not None and current_owner_id != last_owner_id:
                    # Record the pass only if the owning player changed
                    passes.append((last_owner_id, current_owner_id))
                    # Calculate the pass for the new team that received the ball
                    team_passes_counter[current_owner_team] += 1
                
                last_owner_id = current_owner_id
            
            # Coloring the owning player
            if current_owner_id is not None and min_dist < POSSESSION_DISTANCE_THRESHOLD:
                for player_id, box, team_name in players:
                    if player_id == current_owner_id:
                        px1, py1, px2, py2 = box
                        cv2.rectangle(frame, (px1, py1), (px2, py2), color_possession, 4)
                        cv2.putText(frame, f" {team_name} #{player_id} HAS THE BALL",
                                    (px1, py1 - 15), cv2.FONT_HERSHEY_COMPLEX, 0.8, color_possession, 3)

        # Stage Three: Displaying Statistics on the Frame
        start_y = 30
        line_height = 25
        
        # Displaying Team Possession (Most Important)
        offset = start_y
        for team_name, count in team_possession_counter.items():
            team_color_tuple = None
            # Attempt to get the team color for drawing
            for c, name in team_name_map.items():
                if name == team_name:
                    team_color_tuple = c
                    break
            
            # Default draw color if not found
            draw_color = team_color_map.get(team_color_tuple, (255, 255, 255)) 

            # Invert text color if the background is too bright (e.g., if a team color is white/yellow)
            draw_color_text = (0, 0, 0) if np.mean(draw_color) > 180 else (255, 255, 255)
            
            cv2.putText(frame, f"{team_name} Possession: {count} frames",
                        (10, offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3) # Outline
            cv2.putText(frame, f"{team_name} Possession: {count} frames",
                        (10, offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color_text, 2)
            offset += line_height

        cv2.putText(frame, f"Total Passes: {len(passes)}", (10, offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        offset += line_height

        # Displaying player possession (removed from frame drawing to avoid clutter)
        # for idx, (player_id, count) in enumerate(possession_counter.items()):
        #     cv2.putText(frame, f"Player {player_id} Possession: {count} frames",
        #                 (w - 300, start_y + idx * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)


        out.write(frame)

    cap.release()
    out.release()
    
    status_placeholder.text("Analysis complete!")
    
    return output_video_path, possession_counter, team_possession_counter, passes, team_passes_counter

# ----------------------------------------------------------------------
# Streamlit Interface
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="Football Tracking and Analysis - Ultralytics YOLOv8",
    layout="wide"
)

# --- Title and Image Section ---
st.markdown("<h1 style='text-align: center; color: #1F77B4;'>⚽ Ball Possession and Tracking Analyzer (YOLOv8 & Streamlit)</h1>", unsafe_allow_html=True)

# Image insertion as requested
IMAGE_PATH = "football_image.jpg"
if os.path.exists(IMAGE_PATH):
    # Use columns to center the image
    col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
    with col_img2:
        st.image(IMAGE_PATH, use_column_width=True)
else:
    st.warning(f"Image file '{IMAGE_PATH}' not found. Please ensure it is in the same directory.")

st.markdown("Please upload a football match video file to start the tracking process and get possession and passing statistics.")
# -------------------------------


# Warning about the model
st.warning(f"""
    **Important Note:** The model file (weights file) named **`{MODEL_PATH}`** must be placed in the same folder as the `streamlit_app_en.py` file for the application to work.
    """)

# Pre-loading the model
model = load_yolo_model(MODEL_PATH)

if model:
    # 1. Video upload interface
    uploaded_file = st.file_uploader(
        "Upload a video file (preferably .mp4 format)", 
        type=['mp4', 'avi', 'mov']
    )

    if uploaded_file is not None:
        
        # Creating result containers
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Video")
            # Writing the uploaded file temporarily so the CV2 library can read it
            with tempfile.NamedTemporaryFile(delete=False) as tfile_in:
                tfile_in.write(uploaded_file.read())
                video_path_in = tfile_in.name
            
            st.video(video_path_in)

        # Analysis button
        st.markdown("---")
        if st.button("🚀 Start Analysis and Tracking", type="primary"):
            
            # Using a temporary path for the output video
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tfile_out:
                output_path = tfile_out.name
            
            status_placeholder = st.empty()
            progress_bar = st.progress(0, text="Initializing...")

            with st.spinner("Analysis may take some time..."):
                try:
                    # Running the tracking process
                    progress_bar.progress(0.1, text="Loading and setting up the model...")
                    processed_video_path, possession_p, possession_t, passes_list, passes_t = process_video_tracking(
                        video_path_in, model, output_path, status_placeholder
                    )
                    progress_bar.progress(1.0, text="Analysis completed successfully.")

                    # Displaying results in the second column
                    with col2:
                        st.subheader("Processed Video (Tracking and Possession)")
                        st.video(processed_video_path)
                    
                    st.markdown("---")
                    
                    # Displaying Statistics
                    st.header("📊 Summary of Possession and Passing Statistics")

                    # Team Possession Summary
                    if possession_t:
                        team_names = list(possession_t.keys())
                        
                        # Only proceed if we have at least one team identified
                        if team_names:
                            
                            total_frames = sum(possession_t.values())
                            
                            col_a, col_b, col_pass = st.columns(3)

                            # Team A Statistic
                            team_a = team_names[0]
                            col_a.metric(
                                label=f"Possession {team_a}",
                                value=f"{possession_t.get(team_a, 0)} frames",
                                delta=f"{((possession_t.get(team_a, 0) / total_frames) * 100):.2f}% of total" if total_frames > 0 else "0%"
                            )

                            # Team B Statistic (if exists)
                            if len(team_names) > 1:
                                team_b = team_names[1]
                                col_b.metric(
                                    label=f"Possession {team_b}",
                                    value=f"{possession_t.get(team_b, 0)} frames",
                                    delta=f"{((possession_t.get(team_b, 0) / total_frames) * 100):.2f}% of total" if total_frames > 0 else "0%"
                                )
                            else:
                                col_b.metric(label="Possession Team B", value="N/A")

                            # Total Passes
                            col_pass.metric(
                                label="Total Recorded Passes",
                                value=f"{len(passes_list)} passes"
                            )
                        
                        st.subheader("Total Passes per Team (Received by Team):")
                        st.dataframe(
                            {
                                "Team": list(passes_t.keys()),
                                "Number of Passes Received": list(passes_t.values())
                            },
                            hide_index=True
                        )

                    # Player Possession Summary
                    st.subheader("Player Ball Possession Details (Frame Count):")
                    if possession_p:
                        # Converting player statistics to DataFrame and displaying them
                        player_data = []
                        # Create a mapping from player ID to team name based on the discovered team colors
                        # This part attempts to link Player ID back to the team name based on discovery order (A/B)
                        team_id_to_name = {id_val: name for color, name in team_name_map.items() for id_val, team_color in team_colors.items() if tuple(team_color) == color}
                        
                        for player_id, frames in possession_p.items():
                            team_name = team_id_to_name.get(player_id, "N/A")
                            player_data.append({
                                "Player ID": player_id,
                                "Team": team_name,
                                "Possession Frames": frames
                            })
                        
                        st.dataframe(player_data, hide_index=True)
                    else:
                        st.info("No player possession data available (may be due to players not being close enough to the ball).")

                except Exception as e:
                    st.error(f"An error occurred during video processing: {e}")
                finally:
                    # Ensuring temporary files are removed after use
                    if 'video_path_in' in locals() and os.path.exists(video_path_in):
                        os.unlink(video_path_in)
                    if 'output_path' in locals() and os.path.exists(output_path):
                        os.unlink(output_path)
                    progress_bar.empty()
                    status_placeholder.empty()

        else:
             # Cleaning up temporary files when the button is not pressed
             if 'video_path_in' in locals() and os.path.exists(video_path_in):
                os.unlink(video_path_in)
