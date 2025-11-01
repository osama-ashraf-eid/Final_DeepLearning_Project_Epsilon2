import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import io
from collections import defaultdict

# --- 1. CONFIGURATION AND UTILITIES ---

# Define class names based on model training
CLASS_NAMES = {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}

# Path to the trained model (must be verified)
MODEL_PATH = "yolov8m-football_ball_only.pt" 

# Fixed display colors (BGR)
COLOR_BALL = (0, 255, 255) # Yellow/Cyan
COLOR_REFEREE_DISPLAY = (0, 165, 255) # Orange/Amber
COLOR_GOALKEEPER_DISPLAY = (255, 255, 0) # Light yellow for Goalkeeper

# --- New display colors for clarity (BGR) ---
DISPLAY_COLOR_A = (0, 0, 255) # Red for Team A
DISPLAY_COLOR_B = (255, 0, 0) # Blue for Team B
# ---------------------------------------

# Auto-Learning constants
AUTO_LEARNING_FRAMES = 150 # Increase learning samples to 150 frames
# Color tolerance for assignment distance check.
COLOR_TOLERANCE = 120 

# --- UTILITY FOR COLOR ANALYSIS (K-Means Clustering - Pure NumPy) ---

def get_average_color(frame, box):
    """
    Extracts the median BGR color of pixels in the central torso area (1/4 to 1/2 height) 
    of the bounding box for robustness against noise.
    """
    x1, y1, x2, y2 = map(int, box)
    
    # Calculate box height
    h_box = y2 - y1
    
    # Define ROI: Middle section (from 1/4 to 1/2 of the box height)
    y_start = int(y1 + h_box / 4)
    y_end = int(y1 + h_box / 2)
    
    # Clamp y_start/y_end to avoid zero or negative heights
    y_start = max(y1, y_start)
    y_end = min(y2, y_end)
    
    roi = frame[y_start:y_end, x1:x2]
    
    if roi.size == 0:
        # Returns an expected zero NumPy array (float32)
        return np.array([0., 0., 0.], dtype=np.float32)
        
    # Calculate the MEDIAN color in the region for robustness, and return it as float32
    return np.median(roi.reshape(-1,3), axis=0).astype(np.float32)

# Function to perform simplified K-Means clustering using NumPy
def simple_kmeans_numpy(data, k=2, max_iters=10):
    """Simple K-Means clustering implementation using NumPy."""
    
    # Data must be of type float32
    data = data.astype(np.float32) 
    
    if data.shape[0] < k:
        return None

    # 1. Initialize k random centroids
    indices = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[indices]

    for _ in range(max_iters):
        # 2. Assignment step: Find the nearest centroid for each data point
        # Calculate the Euclidean distance for all points in one step
        distances = np.sqrt(np.sum((data - centroids[:, np.newaxis])**2, axis=2))
        labels = np.argmin(distances, axis=0)

        # 3. Update step: Recalculate centroids
        new_centroids = np.array([data[labels == i].mean(axis=0) 
                                 if np.any(labels == i) else centroids[i] 
                                 for i in range(k)])
        
        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break
        
        centroids = new_centroids
        
    return centroids.astype(np.float32)


# Dictionary to permanently store Team A/Team B assignment for Player ID
team_assignment_map = {} 
TEAM_A_CENTER = None # Will be stored as np.ndarray (float32)
TEAM_B_CENTER = None # Will be stored as np.ndarray (float32)


def assign_team_by_reference(player_id, color):
    """
    Assigns the player to Team A or B based on the closest reference color (K-Means Centers).
    """
    global team_assignment_map, TEAM_A_CENTER, TEAM_B_CENTER
    
    # 1. If the player is already assigned, return the saved assignment
    if player_id in team_assignment_map:
        return team_assignment_map[player_id]

    if TEAM_A_CENTER is None or TEAM_B_CENTER is None:
        return "Unassigned" # Classification cannot occur before centers are determined

    # Ensure color_np is float32 to match color centers
    color_np = color.astype(np.float32)
    
    # Calculate Euclidean distance (color distance). Both centers are NumPy arrays
    dist_a = np.linalg.norm(color_np - TEAM_A_CENTER)
    dist_b = np.linalg.norm(color_np - TEAM_B_CENTER)
    
    assigned_team_name = "Unassigned"
    
    # 2. Classification: use only the closest, and use tolerance to check if it's close enough
    if dist_a < dist_b and dist_a < COLOR_TOLERANCE:
        assigned_team_name = "Team A"
    elif dist_b < dist_a and dist_b < COLOR_TOLERANCE:
        assigned_team_name = "Team B"

    # Save assignment if successful
    if assigned_team_name != "Unassigned":
        team_assignment_map[player_id] = assigned_team_name
        
    return assigned_team_name


def determine_team_colors(kit_colors):
    """
    Applies the K-Means algorithm to determine the two color centers (K=2) for the teams.
    """
    global TEAM_A_CENTER, TEAM_B_CENTER
    
    # Samples must be sufficient
    if len(kit_colors) < 50: 
        return 
    
    # Convert the list to a NumPy array of type float32
    colors_np = np.array(kit_colors, dtype=np.float32)
    
    # Apply K-Means
    centers = simple_kmeans_numpy(colors_np, k=2)
    
    if centers is None or centers.shape[0] < 2:
        return 
    
    # Determine Team A and B based on luminosity (Team A is the darker/less luminous)
    # Luminosity is calculated here as the average of BGR values
    luminosity_A_center = np.mean(centers[0])
    luminosity_B_center = np.mean(centers[1])
    
    # Team A is the one with the lowest luminosity (darker)
    if luminosity_A_center < luminosity_B_center:
        # Centers are stored as np.ndarray (float32)
        TEAM_A_CENTER = centers[0]
        TEAM_B_CENTER = centers[1]
    else:
        TEAM_A_CENTER = centers[1]
        TEAM_B_CENTER = centers[0]


# --- 2. CORE PROCESSING LOGIC ---

@st.cache_resource
def load_model():
    """Loads the YOLO model only once and caches it."""
    try:
        st.info(f"Attempting to load YOLO model: {MODEL_PATH}")
        # Prefer using tracking mode to determine ID
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model. Error: {e}")
        st.stop()

def process_video(uploaded_video_file, model):
    
    # Reset global variables on a new run
    global team_assignment_map, TEAM_A_CENTER, TEAM_B_CENTER
    team_assignment_map = {} 
    TEAM_A_CENTER = None
    TEAM_B_CENTER = None
    
    kit_colors_for_learning = [] # List to store colors for the first frames

    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_video_file.read())
        video_path = tfile.name

    output_video_path = os.path.join(tempfile.gettempdir(), "football_tracking_output.mp4")

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Use 'mp4v' for H.264 compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # --- Optimized tracking settings ---
    results = model.track(
        source=video_path,
        conf=0.40,  # Detection confidence
        iou=0.7,    # Intersection Over Union (IoU)
        persist=True,
        tracker="botsort.yaml", 
        stream=True,
        verbose=False
    )
    # -------------------------------------------------------------------

    frame_num = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0, text="Analyzing initial frames for team colors...")
    
    # Dictionary to store determined colors (for final display)
    final_centers_display = {"Team A": None, "Team B": None} 

    for frame_data in results:
        frame_num += 1
        frame = frame_data.orig_img.copy()

        # Update progress bar
        if total_frames > 0:
            progress_bar.progress(min(frame_num / total_frames, 1.0), 
                                  text=f"Processing frame {frame_num} of {total_frames}...")

        # --- Attempt to extract boxes and ID ---
        boxes = None
        classes = None
        ids = None
        
        if frame_data.boxes.id is not None:
            try:
                boxes = frame_data.boxes.xyxy.cpu().numpy()
                classes = frame_data.boxes.cls.cpu().numpy().astype(int)
                ids = frame_data.boxes.id.cpu().numpy().astype(int)
            except Exception:
                pass
        
        # 1. Auto-learning phase
        if frame_num <= AUTO_LEARNING_FRAMES:
            if boxes is not None:
                for box, cls, track_id in zip(boxes, classes, ids):
                    # Ignore ball and referee (cls 0 and 3)
                    if cls in [1, 2]: # Player or Goalkeeper
                        avg_bgr_color = get_average_color(frame, box)
                        kit_colors_for_learning.append(avg_bgr_color)
            
            # If we reached the end of the learning phase, perform calculation
            if frame_num == AUTO_LEARNING_FRAMES:
                 if len(kit_colors_for_learning) >= 50: # Ensure sufficient samples are collected
                     determine_team_colors(kit_colors_for_learning)
                     
                     if TEAM_A_CENTER is not None:
                        # Convert color centers to integer lists for display only
                        final_centers_display["Team A"] = TEAM_A_CENTER.astype(int).tolist()
                        final_centers_display["Team B"] = TEAM_B_CENTER.astype(int).tolist()
                        
                        progress_bar.progress(min(AUTO_LEARNING_FRAMES / total_frames, 1.0), 
                                              text="Color centers successfully determined. Starting tracking...")
                     else:
                         st.warning("Could not determine clear color centers using K-Means. Fallback colors will be used.")
                         # Fallback: Must be a NumPy array
                         TEAM_A_CENTER = np.array([0, 0, 255], dtype=np.float32) 
                         TEAM_B_CENTER = np.array([255, 0, 0], dtype=np.float32)
                         final_centers_display["Team A"] = TEAM_A_CENTER.astype(int).tolist()
                         final_centers_display["Team B"] = TEAM_B_CENTER.astype(int).tolist()
                 else:
                    st.warning("Not enough distinct colors detected in initial frames. Fallback colors will be used.")
                    # Fallback: Must be a NumPy array
                    TEAM_A_CENTER = np.array([0, 0, 255], dtype=np.float32) 
                    TEAM_B_CENTER = np.array([255, 0, 0], dtype=np.float32)
                    final_centers_display["Team A"] = TEAM_A_CENTER.astype(int).tolist()
                    final_centers_display["Team B"] = TEAM_B_CENTER.astype(int).tolist()

        # 2. Tracking and classification phase (after center determination)
        if TEAM_A_CENTER is not None and boxes is not None:
            for box, cls, track_id in zip(boxes, classes, ids):
                x1, y1, x2, y2 = map(int, box)
                track_id_int = int(track_id)

                color = (255, 255, 255) # Default color (white)
                team_label = "Unassigned"

                # --------------------- A. Referee (class 3) ---------------------
                if cls == 3: 
                    team_label = "Referee"
                    color = COLOR_REFEREE_DISPLAY
                
                # ---------------- B. Players and Goalkeepers (class 1, 2) ----------------
                elif cls in [1, 2]:
                    
                    is_goalkeeper = (cls == 1) 
                    
                    # 1. Determine jersey color (returns np.float32 array)
                    avg_bgr_color = get_average_color(frame, (x1, y1, x2, y2))
                    
                    # 2. Assign to team based on extracted color centers (using float arrays)
                    assigned_team_name = assign_team_by_reference(
                        track_id_int, avg_bgr_color
                    )
                    
                    team_label = assigned_team_name

                    # 3. Determine display color based on assignment (using clear colors)
                    if team_label == "Team A":
                        color = DISPLAY_COLOR_A # Using Red for display
                    elif team_label == "Team B":
                        color = DISPLAY_COLOR_B # Using Blue for display
                    else:
                        color = (255, 255, 255) # Unassigned players are white

                    # 4. Color the Goalkeeper with their fixed color
                    if is_goalkeeper and team_label.startswith("Team"):
                        color = COLOR_GOALKEEPER_DISPLAY 
                        team_label = f"GK ({team_label})"
                        
                # --------------------- C. Ball (class 0) ---------------------
                elif cls == 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BALL, 2)
                    cv2.putText(frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BALL, 2)
                    
                # Draw bounding box and ID for all (Players, Goalkeepers, Referees)
                if cls != 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2) 
                    cv2.putText(frame, f"{team_label} ID {track_id_int}", (x1, y1 - 10),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Write frame to output file
        out.write(frame)

    cap.release()
    out.release()
    os.unlink(video_path)

    return output_video_path, final_centers_display


# --- 3. STREAMLIT APP UI (Fully Automated) ---
def streamlit_app():
    # Load the model early and cache it
    model = load_model()

    # 1. Page Config and Background
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
            background-color: rgba(0, 0, 0, 0.7);
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
    st.markdown('<div class="main-title">⚽ Automatic Football Player Detection & Tracking </div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Layout for inputs and preview
    col1, col2 = st.columns([1, 1])
    
    # Global variables must be redeclared here to access their final values after processing
    global TEAM_A_CENTER, TEAM_B_CENTER
    
    uploaded_file = None
    with col1:
        st.subheader("1. Upload Video (Automatic Color Detection) 🎨")
        st.info("Upload an MP4 Video of a Football Match")
        uploaded_file = st.file_uploader("Upload an MP4 Video of a Football Match", type=["mp4"])

        st.markdown("---")
        st.markdown(f"""
            **Team Assignment Logic (Fully Automatic):** The system analyzes the first {AUTO_LEARNING_FRAMES} frames to automatically determine the two main kit colors using the K-Means algorithm.
            - **Team A (Darker Kits):** Displayed in Red ({DISPLAY_COLOR_A} BGR).
            - **Team B (Lighter Kits):** Displayed in Blue ({DISPLAY_COLOR_B} BGR).
            
            *No manual color input is required.*
        """)
        
        st.markdown(f"**Goalkeeper Display Color (Fixed):** {COLOR_GOALKEEPER_DISPLAY}")
        st.markdown(f"**Referee Display Color (Fixed):** {COLOR_REFEREE_DISPLAY}")
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
    if uploaded_file is not None:
        if st.button("Start Tracking & Automatic Team Assignment", key="start_analysis", type="primary"):
            try:
                # Execution
                output_video_path, detected_centers = process_video(uploaded_file, model)

                st.success("Tracking and Classification Complete! 🎉")
                st.markdown("---")

                # --- Output Section ---
                st.subheader("3. Processed Video Output")
                
                # Display the determined colors (optional, for feedback)
                st.markdown(f"""
                    #### Detected Color Centers (BGR - Used for Classification Logic)
                    - **Team A (Darker Kit Center):** `{detected_centers['Team A']}`
                    - **Team B (Lighter Kit Center):** `{detected_centers['Team B']}`
                """)
                
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

            except Exception as e:
                st.error("An error occurred during video processing.")
                st.exception(e)

    elif uploaded_file is None:
        st.info("Upload a video file to enable the analysis.")

if __name__ == '__main__':
    streamlit_app()
