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

# Fixed Display Colors (Roles are visually distinct)
COLOR_BALL = (0, 255, 255) # Yellow/Cyan
COLOR_REFEREE_DISPLAY = (0, 165, 255) # Orange/Amber
COLOR_GOALKEEPER_DISPLAY = (0, 255, 255) # Yellow

# --- UTILITY FOR COLOR ANALYSIS (Auto Clustering Logic) ---

def get_average_color(frame, box):
    """
    يستخلص متوسط لون البيكسلات في الثلث العلوي من صندوق التحديد (القميص) كـ BGR.
    """
    x1, y1, x2, y2 = map(int, box)
    # التركيز على الثلث العلوي كمنطقة القميص لتقليل تأثير العشب
    roi = frame[y1: int(y1 + (y2 - y1) / 3), x1:x2]
    if roi.size == 0:
        return np.array([0,0,0])
    # حساب متوسط اللون في منطقة القميص
    return np.mean(roi.reshape(-1,3), axis=0)

# قاموس لتخزين الألوان المرجعية للفرق التي تم تعيينها
# يتم تعريف هذا المتغير كـ global لإعادة تعيينه في بداية process_video
team_color_references = {} 

def assign_team_by_clustering(player_id, color):
    """
    يقوم بتعيين اللاعب إلى أقرب مجموعة لونية موجودة أو إنشاء مجموعة جديدة إذا كان اللون بعيداً.
    يتبع منطق الكود المرجعي لتقليل عدد مجموعات الألوان (Cluster).
    """
    global team_color_references
    color_np = np.array(color)
    
    # إذا لم يكن اللاعب مُعيّناً بالفعل
    if player_id not in team_color_references:
        
        # 1. إذا كانت هذه أول عملية تعيين، قم بإنشاء المجموعة الأولى
        if not team_color_references:
            # نستخدم ID اللاعب كـ Key مبدئي للقيمة اللونية
            team_color_references[player_id] = color_np 
            return color_np 

        # 2. حاول إيجاد أقرب مجموعة لونية موجودة
        min_dist = 1e9
        closest_player_id = None
        
        for p_id, ref_color in team_color_references.items():
            dist = np.linalg.norm(color_np - ref_color)
            if dist < min_dist:
                min_dist = dist
                closest_player_id = p_id
        
        # 3. التعيين بناءً على التسامح (40 هو حد التسامح كما في الكود المرجعي)
        if min_dist < 40:
            # اللون قريب جداً: قم بتعيين هذا اللاعب إلى نفس مجموعة اللاعب الأقرب
            team_color_references[player_id] = team_color_references[closest_player_id]
        else:
            # اللون بعيد: قم بإنشاء مجموعة لونية جديدة
            team_color_references[player_id] = color_np
            
    return team_color_references[player_id].tolist()


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

def process_video(uploaded_video_file, model):
    
    # إعادة تعيين المتغيرات العالمية في كل عملية جديدة
    global team_color_references
    team_color_references = {} 
    
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

            color = (255, 255, 255) # لون افتراضي (أبيض)
            team_label = "Unassigned"

            # --------------------- A. الحكم (class 3) ---------------------
            if cls == 3: 
                team_label = "Referee"
                color = COLOR_REFEREE_DISPLAY
            
            # ---------------- B. اللاعبون وحراس المرمى (class 1, 2) ----------------
            elif cls in [1, 2]:
                
                is_goalkeeper = (cls == 1) 
                
                # 1. تحديد لون القميص
                avg_bgr_color = get_average_color(frame, (x1, y1, x2, y2))
                
                # 2. التعيين التلقائي للفريق باستخدام منطق التجميع
                # team_color_bgr هي قيمة BGR المرجعية للمجموعة التي ينتمي إليها
                team_color_bgr = assign_team_by_clustering(track_id_int, avg_bgr_color)
                
                # 3. تعيين اسم الفريق ولون العرض بناءً على اللمعان (Luminosity)
                # يتبع المنطق المرجعي (المجموعة الداكنة = Team A, المجموعة الفاتحة = Team B)
                
                # حساب متوسط لمعان اللون (الخفيف/الداكن)
                if np.mean(team_color_bgr) < 128:
                    color = (0, 0, 255) # أحمر/داكن للعرض
                    team_label = "Team A" 
                else:
                    color = (255, 0, 0) # أزرق/فاتح للعرض
                    team_label = "Team B"

                # 4. تلوين الحارس بلونه الخاص (ثابت)
                if is_goalkeeper:
                    color = COLOR_GOALKEEPER_DISPLAY 
                    team_label = f"GK ({team_label})"
                    
            # --------------------- C. الكرة (class 0) ---------------------
            elif cls == 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BALL, 2)
                cv2.putText(frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BALL, 2)
                
            # رسم صندوق التحديد والـ ID للجميع (اللاعبين، الحراس، الحكام)
            if cls != 0:
                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2) 
                 cv2.putText(frame, f"{team_label} ID {track_id_int}", (x1, y1 - 10),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        out.write(frame)

    cap.release()
    out.release()
    os.unlink(video_path)

    return output_video_path


# --- 3. STREAMLIT APP UI (Simplified) ---
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
    st.markdown('<div class="main-title">⚽ Football Tracking: Auto Team Assignment</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Layout for inputs and preview
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Upload Video & Role Display Colors 🎨")
        uploaded_file = st.file_uploader("Upload an MP4 Video of a Football Match", type=["mp4"])

        st.markdown("---")
        st.markdown("""
            **Automatic Team Assignment Logic:** The system automatically clusters player kit colors into two groups.  
            - **Team A (Darker Kit):** Displayed in RED.
            - **Team B (Lighter Kit):** Displayed in BLUE.
        """)
        
        # لا توجد مدخلات لألوان الفريقين هنا، فقط عرض ثابت للألوان
        st.markdown(f"**Goalkeeper Display Color (Yellow):** {COLOR_GOALKEEPER_DISPLAY}")
        st.markdown(f"**Referee Display Color (Orange):** {COLOR_REFEREE_DISPLAY}")
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
                # Execute core logic
                # لا تمرر أي ألوان للفرق، فقط للحارس والحكم (الثابتة)
                output_video_path = process_video(uploaded_file, model)

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
