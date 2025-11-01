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

# Define the class names based on the user's custom model training
CLASS_NAMES = {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}

# Use the user's custom trained model path.
MODEL_PATH = "yolov8m-football_ball_only.pt" 

# --- UTILITY FOR COLOR ANALYSIS ---

def get_average_color(frame, box):
    """
    يستخلص متوسط لون البيكسلات في صندوق التحديد كـ BGR.
    """
    x1, y1, x2, y2 = map(int, box)
    # التركيز على الثلث العلوي كمنطقة القميص
    roi = frame[y1: int(y1 + (y2 - y1) / 3), x1:x2]
    if roi.size == 0:
        return np.array([0,0,0])
    # حساب متوسط اللون في منطقة القميص
    return np.mean(roi.reshape(-1,3), axis=0)

# قاموس لتخزين الألوان المرجعية للفرق التي تم تعيينها
# سنستخدم هذا القاموس عالمياً داخل دالة process_video
team_colors_map = {} 

def assign_team_by_clustering(player_id, color, team_a_bgr, team_b_bgr, BGR_TOLERANCE=55):
    """
    يقوم بتصنيف اللاعب للفريق A أو B بناءً على أقرب مسافة لونية إلى الألوان المرجعية.
    """
    
    color_np = np.array(color)
    team_a_np = np.array(team_a_bgr)
    team_b_np = np.array(team_b_bgr)
    
    # حساب المسافة الإقليدية (مسافة الألوان)
    dist_a = np.linalg.norm(color_np - team_a_np)
    dist_b = np.linalg.norm(color_np - team_b_np)
    
    # التصنيف
    if dist_a < dist_b and dist_a < BGR_TOLERANCE:
        return "Team A", team_a_bgr
    elif dist_b < dist_a and dist_b < BGR_TOLERANCE:
        return "Team B", team_b_bgr
    else:
        # لم يتمكن من التعيين بثقة
        return "Unassigned", (255, 255, 255) # لون أبيض لغير المصنفين


# --- 2. CORE PROCESSING LOGIC ---

@st.cache_resource
def load_model():
    """Loads the YOLO model only once and caches it."""
    try:
        st.info(f"Attempting to load YOLO model: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model. Check MODEL_PATH or network connection. Error: {e}")
        st.stop()

def process_video(uploaded_video_file, model, team_a_bgr, team_b_bgr, goalkeeper_bgr, referee_bgr):
    
    # إعادة تعيين المتغيرات العالمية في كل عملية جديدة
    global team_colors_map
    team_colors_map = {} 
    
    # لا حاجة لعدادات الاستحواذ أو التمريرات

    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_video_file.read())
        video_path = tfile.name

    output_video_path = os.path.join(tempfile.gettempdir(), "football_tracking_output.mp4")

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Use 'mp4v' for H.264 compatibility which is widely supported
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # --- إعدادات التتبع ---
    results = model.track(
        source=video_path,
        conf=0.35, 
        iou=0.6,
        persist=True,
        tracker="botsort.yaml", 
        stream=True,
        verbose=False
    )
    # -------------------------------------------------------------------

    frame_num = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0, text="Processing Frames...")
    
    # ألوان العرض الثابتة
    color_ball = (0, 255, 255)
    color_referee_display = referee_bgr 

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
            out.write(frame)
            continue

        for box, cls, track_id in zip(boxes, classes, ids):
            x1, y1, x2, y2 = map(int, box)
            track_id_int = int(track_id)

            color = (255, 255, 255) # لون افتراضي
            team_label = "Unassigned"

            # --------------------- A. الحكم (class 3) ---------------------
            if cls == 3: 
                team_label = "Referee"
                color = color_referee_display
            
            # ---------------- B. اللاعبون وحراس المرمى (class 1, 2) ----------------
            elif cls in [1, 2]:
                
                is_goalkeeper = (cls == 1) 
                
                # 1. تحديد لون القميص 
                avg_bgr_color = get_average_color(frame, (x1, y1, x2, y2))
                
                # 2. تعيين الفريق وحفظه
                assigned_team_name, assigned_team_color = assign_team_by_clustering(
                    track_id_int, avg_bgr_color, team_a_bgr, team_b_bgr
                )
                
                # حفظ التعيين في القاموس الدائم (للحفاظ على الهوية)
                if assigned_team_name != "Unassigned":
                    team_colors_map[track_id_int] = assigned_team_name
                
                # تحديث التسمية واللون بناءً على ما تم حفظه أو تعيينه الآن
                if track_id_int in team_colors_map:
                    team_label = team_colors_map[track_id_int]
                    color = team_a_bgr if team_label == "Team A" else team_b_bgr
                else:
                    team_label = assigned_team_name
                    color = assigned_team_color
                
                # 3. تلوين الحارس بلونه الخاص
                if is_goalkeeper and team_label.startswith("Team"):
                    color = goalkeeper_bgr 
                    team_label = f"GK ({team_label})"
                    
            # --------------------- C. الكرة (class 0) ---------------------
            elif cls == 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_ball, 2)
                cv2.putText(frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_ball, 2)
                
            # رسم صندوق التحديد والـ ID للجميع (اللاعبين، الحراس، الحكام)
            if cls != 0:
                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2) 
                 cv2.putText(frame, f"{team_label} ID {track_id_int}", (x1, y1 - 10),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # -------------------------------------------------------------------
            
            # لا يوجد منطق للاستحواذ هنا

        out.write(frame)

    cap.release()
    out.release()
    os.unlink(video_path)

    # نكتفي بإرجاع مسار الفيديو المعالج فقط
    return output_video_path


# --- 3. STREAMLIT APP UI ---
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
    st.markdown('<div class="main-title">⚽ Football Detection & Tracking (Color-Based Team Split)</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Layout for inputs and preview
    col1, col2 = st.columns([1, 1])
    
    # دالة مساعدة لتحويل مدخلات BGR
    def parse_bgr(bgr_str, default_bgr):
        try:
            parts = [int(p.strip()) for p in bgr_str.split(',') if p.strip().isdigit()]
            # يجب أن تكون الأجزاء 3 (B, G, R) وقيمها بين 0 و 255
            if len(parts) == 3 and all(0 <= p <= 255 for p in parts):
                return tuple(parts)
            return default_bgr
        except ValueError:
            return default_bgr

    with col1:
        st.subheader("1. Upload Video & Color Configuration 🎨")
        uploaded_file = st.file_uploader("Upload an MP4 Video of a Football Match", type=["mp4"])

        st.markdown("---")
        st.markdown("**Team Colors (BGR Format: Blue, Green, Red)**")
        
        # 1. Team A (Default: White - 255, 255, 255)
        team_a_bgr_str = st.text_input(
            "Team A Color (Used for Auto-Classification - White Kit)",
            value="255, 255, 255", 
        )
        # 2. Team B (Default: Red - 0, 0, 255)
        team_b_bgr_str = st.text_input(
            "Team B Color (Used for Auto-Classification - Red Kit)",
            value="0, 0, 255", 
        )
        
        st.markdown("---")
        st.markdown("**Special Roles Display Colors (BGR)**")

        # 3. Goalkeeper (Default: Yellow - 0, 255, 255)
        goalkeeper_bgr_str = st.text_input(
            "Goalkeeper Display Color (Override Team Color - Yellow)",
            value="0, 255, 255", 
        )

        # 4. Referee (Default: Orange/Amber - 0, 165, 255)
        referee_bgr_str = st.text_input(
            "Referee Display Color (Orange)",
            value="0, 165, 255", 
        )

        # Apply parsing
        team_a_bgr = parse_bgr(team_a_bgr_str, (255, 255, 255))
        team_b_bgr = parse_bgr(team_b_bgr_str, (0, 0, 255))
        goalkeeper_bgr = parse_bgr(goalkeeper_bgr_str, (0, 255, 255))
        referee_bgr = parse_bgr(referee_bgr_str, (0, 165, 255))
        
        st.markdown(f"Configured Colors: **Team A: {team_a_bgr}**, **Team B: {team_b_bgr}**, **GK: {goalkeeper_bgr}**, **Referee: {referee_bgr}**")
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
        if st.button("Start Tracking & Team Classification", key="start_analysis", type="primary"):
            try:
                # Execute core logic
                output_video_path = process_video(
                    uploaded_file, model, team_a_bgr, team_b_bgr, goalkeeper_bgr, referee_bgr
                )

                st.success("Tracking and Classification Complete! 🎉")
                st.markdown("---")

                # --- Output Section ---
                st.subheader("3. Processed Video Output")
                
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
