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
COLOR_BALL = (0, 255, 255)       # Yellow/Cyan - الكرة
COLOR_REFEREE_DISPLAY = (255, 0, 255) # Magenta/Fuchsia - الحكم
COLOR_GOALKEEPER_DISPLAY = (0, 255, 0) # Green - الحارس

# --- NEW DISPLAY COLORS FOR CLARITY ---
DISPLAY_COLOR_A = (0, 0, 255) # Red for Team A (Darker)
DISPLAY_COLOR_B = (255, 0, 0) # Blue for Team B (Lighter)
# ---------------------------------------

# Constants for Auto-Learning
FAST_LEARNING_FRAMES = 5  # التعلم الأولي السريع
AUTO_LEARNING_FRAMES = 100 # التعلم النهائي لجمع عينات أكثر
BGR_TOLERANCE = 100 # *تعديل*: زيادة التسامح لتحسين الفصل اللوني
# CONSTANTS FOR BALL PROXIMITY
BALL_PROXIMITY_FACTOR = 3.0 # *تعديل*: اعتبار اللاعب يمتلك الكرة إذا كانت المسافة أقل من 3.0 مرات من عرض صندوق الكرة.

# --- UTILITY FOR COLOR ANALYSIS (K-Means Clustering - Pure NumPy) ---

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

# Function to perform simplified K-Means clustering using NumPy
def simple_kmeans_numpy(data, k=2, max_iters=10):
    """Simple K-Means clustering implementation using NumPy."""
    
    if data.shape[0] < k:
        return None

    indices = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[indices]

    for _ in range(max_iters):
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        new_centroids = np.array([data[labels == i].mean(axis=0) 
                                     if np.any(labels == i) else centroids[i] 
                                     for i in range(k)])
        
        if np.allclose(centroids, new_centroids):
            break
        
        centroids = new_centroids
        
    return centroids


# قاموس لتخزين تعيين الفريق (Team A/Team B) لـ ID اللاعب بشكل ثابت
team_assignment_map = {} 
TEAM_A_CENTER = None
TEAM_B_CENTER = None

def assign_team_by_reference(player_id, color):
    """
    يقوم بتعيين اللاعب للفريق A أو B بناءً على أقرب لون مرجعي (K-Means Centers).
    """
    global team_assignment_map, TEAM_A_CENTER, TEAM_B_CENTER
    
    # 1. إذا كان اللاعب مُعيّناً بالفعل، أعد التعيين المحفوظ
    if player_id in team_assignment_map:
        return team_assignment_map[player_id]

    if TEAM_A_CENTER is None or TEAM_B_CENTER is None:
        return "Unassigned" # لا يمكن التصنيف قبل تحديد المراكز

    color_np = np.array(color)
    
    # حساب المسافة الإقليدية (مسافة الألوان)
    dist_a = np.linalg.norm(color_np - TEAM_A_CENTER)
    dist_b = np.linalg.norm(color_np - TEAM_B_CENTER)
    
    assigned_team_name = "Unassigned"
    
    # التصنيف بناءً على أقرب مركز لون مرجعي
    if dist_a < dist_b and dist_a < BGR_TOLERANCE:
        assigned_team_name = "Team A"
    elif dist_b < dist_a and dist_b < BGR_TOLERANCE: 
        assigned_team_name = "Team B"

    # حفظ التعيين إذا نجح
    if assigned_team_name != "Unassigned":
        team_assignment_map[player_id] = assigned_team_name
        
    return assigned_team_name


def determine_team_colors(kit_colors, is_final=False):
    """
    يطبق خوارزمية K-Means لتحديد مركزي اللون (K=2) للفريقين.
    """
    global TEAM_A_CENTER, TEAM_B_CENTER
    
    if not is_final and TEAM_A_CENTER is not None:
        return

    if len(kit_colors) < 10: 
        return 
    
    colors_np = np.array(kit_colors, dtype=np.float32)
    
    centers = simple_kmeans_numpy(colors_np, k=2)
    
    if centers is None or centers.shape[0] < 2:
        return 
    
    # تحديد الفريق A و B بناءً على اللمعان (الفريق A هو الداكن)
    luminosity_A = np.mean(centers[0])
    luminosity_B = np.mean(centers[1])
    
    if luminosity_A < luminosity_B:
        center_a, center_b = centers[0], centers[1]
    else:
        center_a, center_b = centers[1], centers[0]

    # حفظ المراكز
    TEAM_A_CENTER = center_a.astype(int).tolist()
    TEAM_B_CENTER = center_b.astype(int).tolist()


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
    
    # إعادة تعيين المتغيرات العالمية
    global team_assignment_map, TEAM_A_CENTER, TEAM_B_CENTER
    team_assignment_map = {} 
    TEAM_A_CENTER = None
    TEAM_B_CENTER = None
    
    kit_colors_for_learning = [] 

    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_video_file.read())
        video_path = tfile.name

    output_video_path = os.path.join(tempfile.gettempdir(), "football_tracking_output.mp4")

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # إعدادات التتبع - *تعديل IOU لزيادة استقرار الـ ID*
    results_iterator = model.track(
        source=video_path,
        conf=0.40,  
        iou=0.8,     # *تم التعديل*: رفع IOU
        persist=True,
        tracker="botsort.yaml", 
        stream=True, 
        verbose=False
    )

    frame_num = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0, text="Analyzing initial frames for team colors...")
    
    
    for frame_data in results_iterator:
        frame_num += 1
        frame = frame_data.orig_img.copy()

        # Update progress bar
        if total_frames > 0:
            progress_bar.progress(min(frame_num / total_frames, 1.0))

        # --- استخلاص الصناديق ---
        boxes, classes, ids = None, None, None
        
        if frame_data.boxes.id is not None:
            try:
                boxes = frame_data.boxes.xyxy.cpu().numpy()
                classes = frame_data.boxes.cls.cpu().numpy().astype(int)
                ids = frame_data.boxes.id.cpu().numpy().astype(int)
            except Exception:
                pass
        
        # متغيرات التخزين المؤقتة
        current_players = [] 
        current_ball = None 
        
        if boxes is not None:
            for box, cls, track_id in zip(boxes, classes, ids):
                x1, y1, x2, y2 = map(int, box)
                track_id_int = int(track_id)
                
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                if cls == 0: # الكرة
                    ball_width = x2 - x1
                    current_ball = (center_x, center_y, box, ball_width)
                elif cls in [1, 2, 3]: # لاعب، حارس، حكم
                    
                    player_center_pos = (center_x, center_y) # *تعديل*: استخدام مركز اللاعب
                    
                    current_players.append({
                        'id': track_id_int,
                        'cls': cls,
                        'box': box,
                        'center_pos': player_center_pos,
                        'kit_center': get_average_color(frame, box)
                    })


        # 1. مرحلة التعلم التلقائي السريع والنهائي (بدون تغيير)
        for player in current_players:
            if player['cls'] in [1, 2] and frame_num <= AUTO_LEARNING_FRAMES: 
                kit_colors_for_learning.append(player['kit_center'])
        
        if frame_num == FAST_LEARNING_FRAMES and len(kit_colors_for_learning) >= 10:
             determine_team_colors(kit_colors_for_learning, is_final=False)
             progress_bar.progress(min(FAST_LEARNING_FRAMES / total_frames, 1.0), 
                                  text="Fast color centers determined. Starting tracking...")
        
        if frame_num == AUTO_LEARNING_FRAMES and len(kit_colors_for_learning) >= 50:
            determine_team_colors(kit_colors_for_learning, is_final=True)
            progress_bar.progress(min(AUTO_LEARNING_FRAMES / total_frames, 1.0), 
                                 text="Final color centers determined. Starting tracking...")
        elif frame_num == AUTO_LEARNING_FRAMES and (TEAM_A_CENTER is None or TEAM_B_CENTER is None):
             st.warning("Not enough distinct colors detected. Using fallback colors (Red/Blue).")
             TEAM_A_CENTER = [0, 0, 255] 
             TEAM_B_CENTER = [255, 0, 0] 
        
        
        # 2. تحديد اللاعب الذي يمتلك الكرة - *منطق محسّن*
        player_with_ball_id = None
        if TEAM_A_CENTER is not None and current_ball is not None:
            ball_pos = np.array(current_ball[0:2])
            ball_width = current_ball[3]
            min_dist = float('inf')

            # تحديد المسافة القصوى المقبولة بناءً على حجم الكرة
            max_allowed_distance = ball_width * BALL_PROXIMITY_FACTOR

            # البحث عن أقرب لاعب للكرة
            for player in current_players:
                if player['cls'] in [1, 2]: # فقط اللاعبين والحراس
                    player_center_pos = np.array(player['center_pos'])
                    # *تعديل*: حساب المسافة بين مركز الكرة ومركز اللاعب
                    distance = np.linalg.norm(player_center_pos - ball_pos)
                    
                    if distance < min_dist:
                        min_dist = distance
                        player_with_ball_id = player['id']

            # تحديد اللاعب الذي يمتلك الكرة (بناءً على المسافة النسبية)
            if min_dist > max_allowed_distance:
                player_with_ball_id = None 


        # 3. رسم الصناديق والبيانات
        for player in current_players:
            box = player['box']
            cls = player['cls']
            track_id_int = player['id']
            avg_bgr_color = player['kit_center']
            x1, y1, x2, y2 = map(int, box)

            color = (255, 255, 255) 
            team_label = "Unassigned"

            # --------------------- A. الحكم (class 3) ---------------------
            if cls == 3: 
                team_label = "Referee"
                color = COLOR_REFEREE_DISPLAY
            
            # ---------------- B. اللاعبون وحراس المرمى (class 1, 2) ----------------
            elif cls in [1, 2]:
                
                is_goalkeeper = (cls == 1) 
                
                if TEAM_A_CENTER is not None:
                    assigned_team_name = assign_team_by_reference(
                        track_id_int, avg_bgr_color
                    )
                    team_label = assigned_team_name
                
                if team_label == "Team A":
                    color = DISPLAY_COLOR_A 
                elif team_label == "Team B":
                    color = DISPLAY_COLOR_B 
                else:
                    color = (255, 255, 255) 

                if is_goalkeeper and team_label.startswith("Team"):
                    color = COLOR_GOALKEEPER_DISPLAY 
                    team_label = f"GK ({team_label.split(' ')[1]})"
                    
                # *التعديل المطلوب*: إضافة نص "has a ball"
                if track_id_int == player_with_ball_id:
                    ball_text = "(has a ball)"
                    # رسم النص فوق صندوق التحديد مباشرة
                    cv2.putText(frame, ball_text, (x1, y1 - 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
            # رسم صندوق التحديد والـ ID للجميع 
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2) 
            cv2.putText(frame, f"{team_label} ID {track_id_int}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # رسم الكرة 
        if current_ball:
            ball_x1, ball_y1, ball_x2, ball_y2 = map(int, current_ball[2])
            cv2.rectangle(frame, (ball_x1, ball_y1), (ball_x2, ball_y2), COLOR_BALL, 2)
            cv2.putText(frame, "Ball", (ball_x1, ball_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BALL, 2)
            
        out.write(frame)

    cap.release()
    out.release()
    os.unlink(video_path)

    return output_video_path


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
    st.markdown('<div class="main-title">⚽ Football Detection & Tracking (Final Optimization) </div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Layout for inputs and preview
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Upload Video (Automatic Color Detection) 🎨")
        uploaded_file = st.file_uploader("Upload an MP4 Video of a Football Match", type=["mp4"])

        st.markdown("---")
        st.markdown(f"""
            #### ✅ Optimization Updates:
            * **Tracking Stability:** IOU increased to 0.8 and Color Tolerance increased to 100 to improve ID stability.
            * **Ball Possession Logic:** Now uses **relative distance** (center-to-center) to dynamically account for perspective. Possession is confirmed if the distance is less than **{BALL_PROXIMITY_FACTOR} times the width of the ball**.
            * **Initial Classification:** Starts after **{FAST_LEARNING_FRAMES} frames** (less than 0.5s).
        """)
        
        st.markdown(f"""
            #### 🌈 Display Colors:
            - Team A: **Red**
            - Team B: **Blue**
            - Goalkeeper: **Green**
            - Referee: **Magenta**
        """)
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
        if st.button("Start Tracking & Optimized Team Assignment", key="start_analysis", type="primary"):
            try:
                # Execute core logic 
                output_video_path = process_video(uploaded_file, model)

                st.success("Tracking and Classification Complete! 🎉")
                st.markdown("---")

                # --- Output Section ---
                st.subheader("3. Processed Video Output")
                
                # Display the determined colors (optional, for feedback)
                st.markdown(f"""
                    #### Detected Color Centers (BGR - Used for Classification Logic)
                    - *Team A (Darker Kit Center):* {TEAM_A_CENTER}
                    - *Team B (Lighter Kit Center):* {TEAM_B_CENTER}
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
