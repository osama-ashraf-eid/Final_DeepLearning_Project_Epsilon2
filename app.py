import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from collections import defaultdict

# --- Configuration & Colors ---
names = {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
color_ball = (0, 255, 255)
color_referee = (200, 200, 200)
color_possession = (0, 255, 0)

# Path to your custom model (ensure this file is present in your deployment)
MODEL_PATH = "yolov8m-football_ball_only.pt" 
TRACKER_CONFIG = "botsort.yaml"

# --- Model Loading and Caching ---
try:
    from ultralytics import YOLO
    
    @st.cache_resource
    def load_model():
        """Loads the YOLO model and handles potential initialization errors."""
        try:
            return YOLO(MODEL_PATH)
        except Exception as e:
            st.error(f"Failed to load YOLO model from {MODEL_PATH}. Check file existence and dependencies (omegaconf, lap).")
            st.code(f"Error: {e}")
            return None

    model = load_model()

except ImportError:
    st.error("Required libraries (ultralytics, collections) not found. Please ensure all dependencies are installed.")
    model = None
except Exception as e:
    st.error(f"An error occurred during model initialization: {e}")
    model = None

# --- Utility Functions (for team assignment) ---

def get_average_color(frame, box):
    """Calculates the average BGR color of a bounding box ROI."""
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0 or x2 <= x1 or y2 <= y1:
        return np.array([0,0,0])
    return np.mean(roi.reshape(-1,3), axis=0)

# Global map to track persistent team colors for IDs
team_colors_map = {} 

def assign_team(player_id, color):
    """Assigns a player ID to a team based on jersey color clustering."""
    global team_colors_map
    
    if player_id not in team_colors_map:
        if not team_colors_map:
            # First player detected sets the first team's color standard
            team_colors_map[player_id] = color
        else:
            min_dist = 1e9
            assigned_team_id = None
            
            # Find the closest existing team color standard
            for pid, c in team_colors_map.items():
                if not isinstance(c, np.ndarray): continue # Skip if already assigned a string name
                dist = np.linalg.norm(color - c)
                if dist < min_dist:
                    min_dist = dist
                    assigned_team_id = pid
            
            # If color is close enough (threshold 40), assign to that team
            if assigned_team_id is not None and min_dist < 40:
                team_colors_map[player_id] = team_colors_map[assigned_team_id]
            else:
                # Else, this player establishes a new team color standard
                team_colors_map[player_id] = color
                
    return team_colors_map[player_id]

# --- Main Processing Function (Contains the core logic) ---

def process_video(video_path, model):
    """
    Runs the YOLO tracking and draws results on the video frames based on the provided logic.
    Returns the path to the output video and the statistics.
    """
    if model is None:
        return None, {}

    # Initialize stats counters
    last_owner_id = None
    possession_counter = defaultdict(int)
    passes = []
    team_possession_counter = defaultdict(int)
    team_passes_counter = defaultdict(int)
    
    # Video setup
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error opening video file: {video_path}")
        return None, {}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use tempfile for the output video
    # FINAL ATTEMPT: Use .avi extension with XVID codec for maximum compatibility
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp_out_file: 
        output_path = tmp_out_file.name

    # CRITICAL CHANGE: Using 'XVID' codec for better compatibility in Linux/Streamlit environment
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Progress bar and status setup
    progress_bar = st.progress(0)
    frame_count = 0
    
    # --- Tracking Loop based on user's logic ---
    try:
        # Generate tracking results
        results_generator = model.track(
            source=video_path,
            conf=0.4,
            iou=0.5,
            tracker=TRACKER_CONFIG,
            persist=True,
            stream=True,
            verbose=False # Suppress verbose output
        )

        for frame_data in results_generator:
            frame_count += 1
            frame = frame_data.orig_img.copy()

            if frame_data.boxes.id is None:
                out.write(frame)
                progress_bar.progress(min(100, int(frame_count / total_frames * 100)))
                continue

            boxes = frame_data.boxes.xyxy.cpu().numpy()
            classes = frame_data.boxes.cls.cpu().numpy().astype(int)
            ids = frame_data.boxes.id.cpu().numpy().astype(int)

            balls, players = [], []
            current_owner_id = None
            current_owner_team = None

            for box, cls, track_id in zip(boxes, classes, ids):
                x1, y1, x2, y2 = map(int, box)

                if cls == 0:
                    # Ball
                    balls.append((track_id, (x1, y1, x2, y2)))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_ball, 2)
                    cv2.putText(frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color_ball, 2)

                elif cls in [1, 2]:
                    # Player or Goalkeeper
                    avg_color = get_average_color(frame, (x1, y1, x2, y2))
                    team_color = assign_team(track_id, avg_color)
                    
                    # Heuristic for drawing color and team name based on the assigned BGR mean
                    if np.mean(team_color) < 128:
                        draw_color = (0, 0, 255) # Red (Team A)
                        team_name = "Team A"
                    else:
                        draw_color = (255, 0, 0) # Blue (Team B)
                        team_name = "Team B"
                        
                    players.append((track_id, (x1, y1, x2, y2), team_name))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
                    cv2.putText(frame, f"{team_name} ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, draw_color, 2)

                else:
                    # Referee
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_referee, 2)
                    cv2.putText(frame, "Referee", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, color_referee, 2)


            # --- Possession Logic ---
            if len(balls) > 0 and len(players) > 0:
                bx1, by1, bx2, by2 = balls[0][1]
                ball_center = np.array([(bx1 + bx2)/2, (by1+by2)/2])

                min_dist = 1e9
                for player_id, box, team_name in players:
                    px1, py1, px2, py2 = box
                    player_center = np.array([(px1 + px2)/2, (py1+py2)/2])
                    dist = np.linalg.norm(ball_center - player_center)
                    if dist < min_dist:
                        min_dist = dist
                        current_owner_id = player_id
                        current_owner_team = team_name

                # Increased threshold for better possession detection (e.g., 150)
                POSSESSION_THRESHOLD = 150 
                if min_dist < POSSESSION_THRESHOLD:
                    possession_counter[current_owner_id] += 1
                    team_possession_counter[current_owner_team] += 1
                    
                    if last_owner_id is not None and current_owner_id != last_owner_id:
                        passes.append((last_owner_id, current_owner_id))
                        # Passes counted for the RECEIVING team (or the team gaining possession)
                        team_passes_counter[current_owner_team] += 1 
                        
                    last_owner_id = current_owner_id
                else:
                    # If ball is loose, reset last owner to avoid counting a pass when no one is near
                    last_owner_id = None 

            # --- Drawing Possession Highlight and Stats Overlay ---
            if current_owner_id is not None:
                for player_id, box, team_name in players:
                    if player_id == current_owner_id:
                        px1, py1, px2, py2 = box
                        cv2.rectangle(frame, (px1, py1), (px2, py2), color_possession, 4)
                        cv2.putText(frame, f" {team_name} ID:{player_id} HAS THE BALL",
                                    (px1, py1 - 15), cv2.FONT_HERSHEY_COMPLEX, 0.8, color_possession, 3)

            # Draw Stats Overlay on the frame (Keeping simple stats for the UI)
            start_y = 30
            
            # Team Possession
            cv2.putText(frame, "--- TEAM POSSESSION ---", (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            start_y += 25
            
            for team_name, count in team_possession_counter.items():
                cv2.putText(frame, f"{team_name} Frames: {count}",
                            (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                start_y += 25
            
            # Total Passes
            cv2.putText(frame, f"Total Passes: {len(passes)}", (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            start_y += 30

            out.write(frame)
            
            # Update progress
            progress_bar.progress(min(100, int(frame_count / total_frames * 100)))

        progress_bar.empty()
        st.success("Video analysis complete!")
        
    except Exception as e:
        st.error(f"An error occurred during video processing loop: {e}")
        output_path = None
        
    finally:
        cap.release()
        out.release()
        # Clean up temporary output if processing failed (no unlink here, let Streamlit manage it)
        if output_path and not os.path.exists(output_path):
            output_path = None

    stats = {
        'player_possession': possession_counter,
        'team_possession': team_possession_counter,
        'total_passes': len(passes),
        'passes_list': passes,
        'team_passes_counter': team_passes_counter
    }
    return output_path, stats


# --- 4. Streamlit UI Layout (Applied Requested Changes) ---
st.set_page_config(layout="wide", page_title="Football Tracking & Analysis")

# Centered Title
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>⚽ Football Detection & Tracking 📊</h1>", unsafe_allow_html=True)
st.markdown("---")

# Image (Using use_container_width=True as requested for deprecation fix)
st.image("football_img.jpg", use_container_width=True)

st.markdown("---")
uploaded_file = st.file_uploader(
    "Upload a football video (MP4 or MOV)", 
    type=["mp4", "mov", "avi"]
)

if uploaded_file is not None:
    # Save the uploaded file temporarily
    # NOTE: The suffix here is .mp4 for the uploaded file, which is fine.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file: 
        tmp_file.write(uploaded_file.read())
        video_file_path = tmp_file.name

    st.subheader("Original Video Preview")
    st.video(video_file_path)

    # Button to start analysis
    if st.button("Start Tracking & Analysis", use_container_width=True):
        st.info("Processing video...")
        
        # Placeholder for status messages (cleared after analysis)
        status_placeholder = st.empty()
        status_placeholder.info("Loading model and preparing analysis...")
        
        # Run the analysis function
        output_path, stats = process_video(video_file_path, model)

        # 5. Display Results
        if output_path and os.path.exists(output_path):
            
            # --- Display Video Result ---
            st.subheader("Analyzed Video Result")
            st.video(output_path) 
            
            status_placeholder.empty()

            # --- ADD DOWNLOAD BUTTON HERE ---
            try:
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="⬇️ Download Analyzed Video (.avi)",
                        data=file,
                        file_name="analyzed_football_video.avi",
                        mime="video/avi",
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"Could not prepare download link: {e}")

            # --- REMOVED ANALYSIS SUMMARY SECTION HERE AS REQUESTED ---
            
        # Clean up the original uploaded file (output file cleanup is tricky in Streamlit, 
        # so we leave the output_path management as simple as possible)
        if os.path.exists(video_file_path):
            os.unlink(video_file_path)
        
    st.markdown("---")

st.markdown("""
<div style='text-align: center; color: #6B7280; font-size: 0.8rem;'>
    Note: The custom YOLO model must be accessible in your Streamlit cloud environment for analysis to run successfully.
</div>
""", unsafe_allow_html=True)
