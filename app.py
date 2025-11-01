import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import io
from collections import defaultdict
# تم حذف: from sklearn.cluster import KMeans - لتجنب خطأ ModuleNotFoundError

# --- 1. CONFIGURATION AND UTILITIES ---

# Define the class names based on the user's custom model training
CLASS_NAMES = {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}

# Use the user's custom trained model path.
MODEL_PATH = "yolov8m-football_ball_only.pt" 

# Fixed Display Colors (Roles are visually distinct)
COLOR_BALL = (0, 255, 255) # Yellow/Cyan
COLOR_REFEREE_DISPLAY = (0, 165, 255) # Orange/Amber
COLOR_GOALKEEPER_DISPLAY = (0, 255, 255) # Yellow

# Constants for Auto-Learning
AUTO_LEARNING_FRAMES = 50 # عدد الإطارات التي يتم تجميع الألوان منها
BGR_TOLERANCE = 70 # زيادة التسامح قليلاً بسبب التعقيد اللوني بعد التجميع

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
    
    # Check if data size is sufficient
    if data.shape[0] < k:
        return None

    # 1. Initialize k centroids randomly
    indices = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[indices]

    for _ in range(max_iters):
        # 2. Assignment Step: Find the nearest centroid for each data point
        # توسيع Centroids و Data لتتمكن NumPy من حساب المسافات لجميع النقاط دفعة واحدة
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        # 3. Update Step: Recalculate centroids
        new_centroids = np.array([data[labels == i].mean(axis=0) 
                                  if np.any(labels == i) else centroids[i] 
                                  for i in range(k)])
        
        # Check for convergence (small change in centroids)
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


def determine_team_colors(kit_colors):
    """
    يطبق خوارزمية K-Means (باستخدام NumPy) لتحديد مركزي اللون (K=2) للفريقين.
    """
    global TEAM_A_CENTER, TEAM_B_CENTER
    
    if len(kit_colors) < 50: # Adjusting check to be safe
        # st.error("Not enough color samples collected for clustering.") # لا يمكن استخدام Streamlit هنا
        return 
    
    colors_np = np.array(kit_colors, dtype=np.float32)
    
    # تطبيق K-Means باستخدام NumPy بدلاً من scikit-learn
    centers = simple_kmeans_numpy(colors_np, k=2)
    
    if centers is None or centers.shape[0] < 2:
        return 
    
    # تحديد الفريق A و B بناءً على اللمعان (الفريق A هو الداكن)
    luminosity_A = np.mean(centers[0])
    luminosity_B = np.mean(centers[1])
    
    # الفريق A هو صاحب اللمعان الأقل (الداكن)
    if luminosity_A < luminosity_B:
        TEAM_A_CENTER = centers[0]
        TEAM_B_CENTER = centers[1]
    else:
        TEAM_A_CENTER = centers[1]
        TEAM_B_CENTER = centers[0]

    # تحويل المراكز إلى قوائم BGR integers
    TEAM_A_CENTER = TEAM_A_CENTER.astype(int).tolist()
    TEAM_B_CENTER = TEAM_B_CENTER.astype(int).tolist()


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
    
    kit_colors_for_learning = [] # قائمة لتخزين الألوان للإطارات الأولى

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
    progress_bar = st.progress(0, text="Analyzing initial frames for team colors...")
    
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
            
        # 1. مرحلة التعلم التلقائي (أول 50 إطار)
        if frame_num <= AUTO_LEARNING_FRAMES:
            for box, cls, track_id in zip(boxes, classes, ids):
                 # تجاهل الحكم (cls != 3)
                 if cls in [1, 2]: # لاعب أو حارس
                    avg_bgr_color = get_average_color(frame, box)
                    kit_colors_for_learning.append(avg_bgr_color)
            
            # إذا وصلنا لنهاية مرحلة التعلم، قم بالحساب
            if frame_num == AUTO_LEARNING_FRAMES:
                 if len(kit_colors_for_learning) >= 50: # تأكد من جمع عينات كافية
                    determine_team_colors(kit_colors_for_learning)
                    progress_bar.progress(min(AUTO_LEARNING_FRAMES / total_frames, 1.0), 
                                          text="Color centers determined. Starting tracking...")
                 else:
                    st.warning("Not enough distinct colors detected in initial frames. Classification may be inaccurate.")
                    TEAM_A_CENTER = [0, 0, 0] # fallback
                    TEAM_B_CENTER = [255, 255, 255] # fallback
            
            if TEAM_A_CENTER is None: # إذا لم يتم التحديد بعد، استمر في تسجيل الفيديو بدون تصنيف
                out.write(frame)
                continue
            
            
        # 2. مرحلة التتبع والتصنيف (بعد أول 50 إطار)
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
                
                # 2. التعيين للفريق بناءً على المراكز اللونية المستخلصة
                assigned_team_name = assign_team_by_reference(
                    track_id_int, avg_bgr_color
                )
                
                team_label = assigned_team_name

                # 3. تحديد لون العرض بناءً على التعيين (لون المركز المستخلص)
                if team_label == "Team A":
                    color = TEAM_A_CENTER
                elif team_label == "Team B":
                    color = TEAM_B_CENTER
                else:
                    color = (255, 255, 255) # Unassigned players are white

                # 4. تلوين الحارس بلونه الخاص (ثابت)
                if is_goalkeeper and team_label.startswith("Team"):
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
    st.markdown('<div class="main-title">⚽ Football Tracking: Auto-Learning Team Colors</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Layout for inputs and preview
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Upload Video (Automatic Color Detection) 🎨")
        uploaded_file = st.file_uploader("Upload an MP4 Video of a Football Match", type=["mp4"])

        st.markdown("---")
        st.markdown("""
            **Team Assignment Logic (Fully Automatic):** The system analyzes the first 50 frames to automatically determine the two main kit colors using **K-Means Clustering (NumPy based)**.
            - **Team A:** Assigned to the DARKER of the two detected colors.
            - **Team B:** Assigned to the LIGHTER of the two detected colors.
            
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
                # Execute core logic (No color inputs needed)
                output_video_path = process_video(uploaded_file, model)

                st.success("Tracking and Classification Complete! 🎉")
                st.markdown("---")

                # --- Output Section ---
                st.subheader("3. Processed Video Output")
                
                # Display the determined colors (optional, for feedback)
                st.markdown(f"""
                    #### Detected Color Centers (BGR)
                    - **Team A (Darker):** `{TEAM_A_CENTER}`
                    - **Team B (Lighter):** `{TEAM_B_CENTER}`
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
