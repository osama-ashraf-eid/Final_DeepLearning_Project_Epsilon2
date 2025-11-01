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
CLASS_NAMES = {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}

# Use the user's custom trained model path.
# ملاحظة: يجب أن يكون هذا الملف موجودًا في نفس مسار تشغيل التطبيق.
MODEL_PATH = "yolov8m-football_ball_only.pt" 

# --- UTILITY FOR COLOR ANALYSIS ---

def get_kit_color_bgr(frame, box):
    """
    يستخلص متوسط لون الجزء العلوي من صندوق التحديد (القميص).
    """
    x1, y1, x2, y2 = map(int, box)
    
    # خذ الثلث الأوسط من الجزء العلوي (منطقة القميص)
    # نأخذ الثلث الأوسط من الجزء العلوي لتقليل تأثير الخلفية (العشب)
    kit_area = frame[y1 : int(y1 + (y2 - y1) / 3), x1 : x2] 
    
    if kit_area.size == 0:
        return (0, 0, 0) # أسود في حالة الفشل
    
    # حساب متوسط لون BGR في المنطقة المحددة
    avg_color_bgr = np.mean(kit_area, axis=(0, 1)).astype(int).tolist()
    return tuple(avg_color_bgr)

def classify_team_by_color(bgr_color, team_a_bgr, team_b_bgr, BGR_TOLERANCE=55): # تم زيادة التسامح قليلاً للإضاءة
    """
    يصنف اللون بناءً على أقرب مسافة إقليدية في مساحة BGR إلى ألوان الفريقين A و B.
    """
    color_np = np.array(bgr_color)
    team_a_np = np.array(team_a_bgr)
    team_b_np = np.array(team_b_bgr)
    
    # حساب المسافة الإقليدية (مسافة الألوان)
    dist_a = np.linalg.norm(color_np - team_a_np)
    dist_b = np.linalg.norm(color_np - team_b_np)
    
    # التصنيف
    if dist_a < dist_b and dist_a < BGR_TOLERANCE:
        return "Team A"
    elif dist_b < dist_a and dist_b < BGR_TOLERANCE:
        return "Team B"
    else:
        return None # لم يتم التصنيف بثقة
    
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
    # قاموس لحفظ الـ ID والفريق المقابل له بشكل دائم بناءً على تحليل اللون
    player_team_map = {} 

    # --- إعدادات التتبع ---
    results = model.track(
        source=video_path,
        conf=0.35, 
        iou=0.6,
        persist=True,
        tracker="botsort.yaml", # استخدام BoT-SORT لحفظ هوية أفضل
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

        balls, players_for_possession = [], [] # قائمة players_for_possession تشمل اللاعبين والحراس المصنفين فقط

        for box, cls, track_id in zip(boxes, classes, ids):
            x1, y1, x2, y2 = map(int, box)
            track_id_int = int(track_id)

            color = (255, 255, 255) # لون افتراضي أبيض (لغير المصنفين)
            team_label = "Unassigned"

            # --------------------- A. الحكم (class 3) ---------------------
            if cls == 3: 
                team_label = "Referee"
                color = referee_bgr
            
            # ---------------- B. اللاعبون وحراس المرمى (class 1, 2) ----------------
            elif cls in [1, 2]:
                
                is_goalkeeper = (cls == 1) # بناءً على تعريف CLASS_NAMES
                
                # 1. محاولة الحصول على الفريق من القاموس المحفوظ (للحفاظ على الهوية)
                if track_id_int in player_team_map:
                    assigned_team = player_team_map[track_id_int]
                    team_label = assigned_team
                    if team_label == "Team A":
                        color = team_a_bgr
                    else:
                        color = team_b_bgr
                
                # 2. إذا لم يتم التعيين بعد، قم بتحليل اللون والتعيين
                else:
                    kit_bgr = get_kit_color_bgr(frame, box)
                    assigned_team = classify_team_by_color(kit_bgr, team_a_bgr, team_b_bgr)
                    
                    if assigned_team:
                        team_label = assigned_team
                        player_team_map[track_id_int] = assigned_team # حفظ التعيين
                        if team_label == "Team A":
                            color = team_a_bgr
                        else:
                            color = team_b_bgr
                
                # 3. تلوين الحارس بلونه الخاص (إذا تم تصنيفه لفريق)
                if is_goalkeeper and (team_label.startswith("Team A") or team_label.startswith("Team B")):
                    color = goalkeeper_bgr 
                    # نستخدم اسم الفريق الذي تم تصنيف الحارس إليه
                    base_team = "Team A" if "Team A" in team_label else "Team B" 
                    team_label = f"GK ({base_team})"
                
                # إضافة اللاعب/الحارس المصنف لقائمة تحليل الاستحواذ
                if team_label.startswith("Team"):
                    # نرسل التسمية الأساسية للفريق (Team A أو Team B)
                    base_team_label = "Team A" if "Team A" in team_label else "Team B" 
                    players_for_possession.append((track_id_int, (x1, y1, x2, y2), base_team_label))
                    
            # --------------------- C. الكرة (class 0) ---------------------
            elif cls == 0:
                balls.append((track_id_int, (x1, y1, x2, y2)))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
            # رسم صندوق التحديد والـ ID للجميع (اللاعبين، الحراس، الحكام)
            if cls != 0:
                 # استخدام اللون المخصص للفريق/الحارس/الحكم
                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2) 
                 # كتابة اسم الفريق أو الدور بجانب الـ ID
                 cv2.putText(frame, f"{team_label} ID {track_id_int}", (x1, y1 - 10),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # -------------------------------------------------------------------

            # --- حساب الاستحواذ ---
        ball_owner_id = None
        min_dist = None
        possession_detected = False
        POSSESSION_THRESHOLD = 80

        if len(balls) > 0 and len(players_for_possession) > 0:
            # Get ball center (assuming one ball)
            ball_box = balls[0][1]
            bx1, by1, bx2, by2 = ball_box
            ball_center = np.array([(bx1 + bx2) / 2, (by1 + by2) / 2])

            min_dist = 1e9
            for player_id, player_box, team_label in players_for_possession:
                # Get player center
                px1, py1, px2, py2 = player_box
                player_center = np.array([(px1 + px2) / 2, (py1 + py2) / 2])

                # Calculate distance
                dist = np.linalg.norm(ball_center - player_center)

                if dist < min_dist:
                    min_dist = dist
                    ball_owner_id = player_id
            
            # Detect and highlight possession
            if min_dist < POSSESSION_THRESHOLD and ball_owner_id in player_team_map:
                possession_detected = True
                for player_id, player_box, team_label in players_for_possession:
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
            # استخدام القاموس للحصول على اسم الفريق
            owner_team_full = player_team_map.get(ball_owner_id) 
            # تنظيف الاسم في الـ log ليكون "Team A" أو "Team B" فقط
            owner_team = "Team A" if "Team A" in owner_team_full else ("Team B" if "Team B" in owner_team_full else None)

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
        
        # 1. Team A
        team_a_bgr_str = st.text_input(
            "Team A Color (Used for Auto-Classification)",
            value="255, 100, 0", # Default Blue (BGR)
        )
        # 2. Team B
        team_b_bgr_str = st.text_input(
            "Team B Color (Used for Auto-Classification)",
            value="0, 0, 255", # Default Red (BGR)
        )
        
        st.markdown("---")
        st.markdown("**Special Roles Display Colors (BGR)**")

        # 3. Goalkeeper
        goalkeeper_bgr_str = st.text_input(
            "Goalkeeper Display Color (Override Team Color)",
            value="0, 255, 0", # Default Green (BGR)
        )

        # 4. Referee
        referee_bgr_str = st.text_input(
            "Referee Display Color",
            value="0, 255, 255", # Default Yellow (BGR)
        )

        # Apply parsing
        team_a_bgr = parse_bgr(team_a_bgr_str, (255, 100, 0))
        team_b_bgr = parse_bgr(team_b_bgr_str, (0, 0, 255))
        goalkeeper_bgr = parse_bgr(goalkeeper_bgr_str, (0, 255, 0))
        referee_bgr = parse_bgr(referee_bgr_str, (0, 255, 255))
        
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
        if st.button("Start Tracking & Possession Analysis", key="start_analysis", type="primary"):
            try:
                # Execute core logic
                output_video_path, output_csv_path, df_log = process_video(
                    uploaded_file, model, team_a_bgr, team_b_bgr, goalkeeper_bgr, referee_bgr
                )

                st.success("Analysis Complete! 🎉")
                st.markdown("---")

                # --- Output Section ---
                st.subheader("3. Processed Video Output & Data")
                output_col1, output_col2 = st.columns([1, 1])

                with output_col1:
                    st.markdown("#### Annotated Video")
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
                    st.markdown("#### Possession Log (CSV)")
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
                st.error("An error occurred during video processing.")
                st.exception(e)

    elif uploaded_file is None:
        st.info("Upload a video file to enable the analysis.")

if __name__ == '__main__':
    streamlit_app()
