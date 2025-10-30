import streamlit as st
import cv2
import numpy as np
import tempfile
import os

try:
    from ultralytics import YOLO
    from collections import defaultdict
    MODEL_PATH = "yolov8m-football_ball_only.pt"
    @st.cache_resource
    def load_model():
        try:
            return YOLO(MODEL_PATH)
        except Exception as e:
            st.error(f"Failed to load YOLO model from {MODEL_PATH}. Please ensure the file is accessible in the deployment environment.")
            st.code(f"Error: {e}")
            return None

    model = load_model()

except ImportError:
    st.error("Required libraries (ultralytics, collections) not found. Please ensure all dependencies are installed.")
    model = None
except Exception as e:
    st.error(f"An error occurred during model initialization: {e}")
    model = None

names = {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
color_ball = (0, 255, 255)
color_referee = (200, 200, 200)
color_possession = (0, 255, 0)


def get_average_color(frame, box):
    """Calculates the average BGR color of a bounding box ROI."""
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0 or x2 <= x1 or y2 <= y1:
        return np.array([0,0,0])
    return np.mean(roi.reshape(-1,3), axis=0)

team_colors = {}
team_possession_counter = defaultdict(int)


def assign_team(player_id, color, team_colors_map):
    """Assigns a player to a team based on jersey color clustering."""
    
    if player_id not in team_colors_map:
        if len(team_colors_map) == 0:
            team_colors_map[player_id] = color
        else:
            min_dist = 1e9
            assigned_team_id = None
            for pid, c in team_colors_map.items():
                # Check for existing team colors
                if isinstance(c, str): continue # Skip placeholder team names/colors
                dist = np.linalg.norm(color - c)
                if dist < min_dist:
                    min_dist = dist
                    assigned_team_id = pid
            
            if assigned_team_id is not None and min_dist < 40:
                team_colors_map[player_id] = team_colors_map[assigned_team_id]
            else:
                team_colors_map[player_id] = color
                
    return team_colors_map[player_id]


def process_video(video_path, model):
    """
    Runs the YOLO tracking and draws results on the video frames.
    Returns the path to the output video and the statistics.
    """
    if model is None:
        st.warning("Cannot run tracking analysis because the model failed to load.")
        return None, {}

    # Initializing containers for analysis
    last_owner_id = None
    possession_counter = defaultdict(int)
    passes = []
    team_colors_map = {} # To store player ID -> assigned color/team
    team_possession_counter = defaultdict(int)
    team_passes_counter = defaultdict(int)
    
    # Setup video capture and writer
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error opening video file: {video_path}")
        return None, {}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Use tempfile for the output video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_out_file:
        output_path = tmp_out_file.name

    # Use 'avc1' for compatibility with Streamlit/browsers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    st.info(f"Starting analysis on video ({w}x{h} @ {fps:.2f} fps)... This may take a few minutes.")
    
    # Progress bar for the tracking process
    progress_bar = st.progress(0)
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # --- Tracking Loop ---
    
    # Using 'stream=True' is good for efficiency, but Streamlit requires a clear loop
    # We will use the model.track method which returns a generator
    try:
        results_generator = model.track(
            source=video_path,
            conf=0.4,
            iou=0.5,
            tracker="botsort.yaml", # Make sure botsort.yaml is available if needed
            persist=True,
            stream=True,
            # Suppress verbose output during processing
            verbose=False
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
            
            # 1. Detection and Team Assignment
            for box, cls, track_id in zip(boxes, classes, ids):
                x1, y1, x2, y2 = map(int, box)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                if cls == 0:
                    # Ball
                    balls.append((track_id, (x1, y1, x2, y2)))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_ball, 2)
                    cv2.putText(frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color_ball, 2)

                elif cls in [1, 2]:
                    # Player or Goalkeeper
                    avg_color = get_average_color(frame, (x1, y1, x2, y2))
                    
                    # Ensure the team color is set only if the player is new
                    if track_id not in team_colors_map:
                        assign_team(track_id, avg_color, team_colors_map)

                    # Determine team draw color and name based on the assigned color (simple BGR clustering)
                    team_color_bgr = team_colors_map.get(track_id, (0, 0, 0))
                    
                    # Simple heuristic: dark color vs light color
                    if np.mean(team_color_bgr) < 128:
                        draw_color = (0, 0, 255) # Red (Team A)
                        team_name = "Team A"
                    else:
                        draw_color = (255, 0, 0) # Blue (Team B)
                        team_name = "Team B"
                        
                    players.append((track_id, (x1, y1, x2, y2), team_name))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
                    
                    # Draw team name and ID
                    label = f"{team_name} ID:{track_id} ({names[cls]})"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, draw_color, 2)

                else:
                    # Referee
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_referee, 2)
                    cv2.putText(frame, names.get(cls, "Ref/Other"), (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, color_referee, 2)

            # 2. Possession Tracking and Pass Counting
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

                # 90 is the distance threshold for possession (in pixels)
                if min_dist < 90 and current_owner_id is not None:
                    possession_counter[current_owner_id] += 1
                    team_possession_counter[current_owner_team] += 1

                    if last_owner_id is not None and current_owner_id != last_owner_id:
                        passes.append((last_owner_id, current_owner_id, last_owner_id in [p[0] for p in players] and current_owner_id in [p[0] for p in players]))
                        team_passes_counter[current_owner_team] += 1
                        
                    last_owner_id = current_owner_id

            # 3. Draw Possession Highlight and Text Overlay
            if current_owner_id is not None:
                for player_id, box, team_name in players:
                    if player_id == current_owner_id:
                        px1, py1, px2, py2 = box
                        cv2.rectangle(frame, (px1, py1), (px2, py2), color_possession, 4)
                        cv2.putText(frame, f" {team_name} ID:{player_id} HAS THE BALL",
                                    (px1, py1 - 15), cv2.FONT_HERSHEY_COMPLEX, 0.8, color_possession, 3)

            # Draw Stats Overlay on the frame
            start_y = 30
            # Team Possession Stats
            cv2.putText(frame, "--- TEAM STATS ---", (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            start_y += 25
            for team_name, count in team_possession_counter.items():
                cv2.putText(frame, f"{team_name} Possession: {count} frames",
                            (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                start_y += 25
            
            # Total Passes
            cv2.putText(frame, f"Total Passes: {len(passes)}", (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            start_y += 30
            
            # Player Possession Stats
            cv2.putText(frame, "--- PLAYER POSSESSION ---", (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            start_y += 25
            
            for player_id, count in sorted(possession_counter.items()):
                # Determine team for display (simplified for overlay text)
                current_team_name = next((p[2] for p in players if p[0] == player_id), "Unknown")
                
                cv2.putText(frame, f"{current_team_name} ID:{player_id}: {count} frames",
                            (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                start_y += 20
                
            out.write(frame)
            
            # Update progress
            progress_bar.progress(min(100, int(frame_count / total_frames * 100)))

        progress_bar.empty()
        st.success("Video analysis complete!")
        
    except Exception as e:
        st.error(f"An error occurred during video processing. Please check the model and file paths: {e}")
        output_path = None
        
    finally:
        cap.release()
        out.release()
        # Clean up temporary output if processing failed
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


# 4. Streamlit UI Layout
st.set_page_config(layout="wide", page_title="Football Tracking & Analysis")


st.title("Football detection & tracking")
st.markdown("---")
st.image("football_img.jpg", use_column_width=True)

st.markdown("---")
uploaded_file = st.file_uploader(
    "Upload a football video (MP4 or MOV)", 
    type=["mp4", "mov", "avi"]
)

if uploaded_file is not None:
    # حفظ الملف المؤقت
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_file_path = tmp_file.name

    st.subheader("Original Video Preview")
    st.video(video_file_path)

    # زر لبدء التحليل
    if st.button("Start Tracking & Analysis"):
        st.info("Processing video...")
        
        # حاوية لإظهار شريط التقدم والرسائل
        status_placeholder = st.empty()
        status_placeholder.info("Loading model and preparing analysis...")
        
        # تشغيل الدالة
        output_path, stats = process_video(video_file_path, model)

        # 5. عرض النتائج
        if output_path and os.path.exists(output_path):
            st.subheader("Analyzed Video Result")
            st.video(output_path)
            
            status_placeholder.empty()

            st.markdown("## Analysis Summary")
            
            col1, col2 = st.columns(2)
            
            # عرض إحصائيات الفرق
            with col1:
                st.subheader("Team Possession (Frames Count)")
                team_data = {
                    "Team": list(stats['team_possession'].keys()),
                    "Possession Frames": list(stats['team_possession'].values())
                }
                st.dataframe(team_data, use_container_width=True, hide_index=True)
                
                total_passes = stats['total_passes']
                st.metric("Total Passes Detected", total_passes)

                st.subheader("Passes per Team")
                team_pass_data = {
                    "Team": list(stats['team_passes_counter'].keys()),
                    "Passes": list(stats['team_passes_counter'].values())
                }
                st.dataframe(team_pass_data, use_container_width=True, hide_index=True)


            # عرض إحصائيات اللاعبين
            with col2:
                st.subheader("Player Possession (Frames Count)")
                player_data = []
                for player_id, count in stats['player_possession'].items():
                    player_data.append({
                        "Player ID": player_id,
                        "Possession Frames": count
                    })
                st.dataframe(player_data, use_container_width=True, hide_index=True)
                
                # عرض قائمة التمريرات
                st.subheader("Pass List")
                pass_list_data = []
                for i, (from_id, to_id, _) in enumerate(stats['passes_list'], 1):
                    pass_list_data.append({
                        "#": i,
                        "Pass": f"ID {from_id} -> ID {to_id}"
                    })
                st.dataframe(pass_list_data, use_container_width=True, hide_index=True)
                
        # تنظيف الملف المؤقت للفيديو المُعالج
        if output_path and os.path.exists(output_path):
            os.unlink(output_path)

    # تنظيف الملف المؤقت للفيديو الأصلي
    if os.path.exists(video_file_path):
        os.unlink(video_file_path)
        
    st.markdown("---")

st.markdown("""
<div style='text-align: center; color: #6B7280; font-size: 0.8rem;'>
    Note: The custom YOLO model must be accessible in your Streamlit cloud environment for analysis to run successfully.
</div>
""", unsafe_allow_html=True)
