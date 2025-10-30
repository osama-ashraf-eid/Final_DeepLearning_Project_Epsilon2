import os
import sys
import subprocess

# ✅ تثبيت OpenCV في حال عدم وجوده
# ملاحظة: تم تحديده بالفعل في requirements.txt، ولكن هذا التأكد جيد.
try:
    import cv2
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python-headless==4.8.0.74"])
    import cv2

import streamlit as st
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import tempfile



# --------------------- PAGE SETUP ---------------------
# تعيين تخطيط الصفحة
st.set_page_config(page_title="Football Tracking & Analysis", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; color: #00BFFF;'>⚽ Football Match Tracking & Analysis</h1>
    """,
    unsafe_allow_html=True
)

# --------------------- LOCAL IMAGE DISPLAY ---------------------
IMAGE_PATH = "football_img.jpg"
if os.path.exists(IMAGE_PATH):
    # تم تغيير use_container_width=True إلى width='stretch' لحل تحذير Streamlit
    st.image(IMAGE_PATH, width='stretch', caption="Automated Football Match Analysis using YOLOv8")
else:
    st.warning(f"⚠️ Image file not found: {IMAGE_PATH}")

st.write("---")

# --------------------- MODEL PATH ---------------------
MODEL_PATH = "yolov8m-football_ball_only.pt"
if not os.path.exists(MODEL_PATH):
    # يمكنك استخدام gdown هنا لتحميل ملفات النماذج الكبيرة من Google Drive إذا كانت عامة
    st.error(f"❌ Model file not found: {MODEL_PATH}")
    st.stop()

# Load YOLO model once
model = YOLO(MODEL_PATH)
st.success("✅ Model loaded successfully!")

# --------------------- FILE UPLOAD ---------------------
uploaded_video = st.file_uploader("🎥 Upload a football video (.mp4)", type=["mp4"])

if uploaded_video:
    # حفظ الفيديو المُحمل في ملف مؤقت
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        tmp_video.write(uploaded_video.read())
        video_path = tmp_video.name

    st.info("Video uploaded successfully. Starting analysis...")

    # --------------------- OUTPUT VIDEO SETUP ---------------------
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    # استخدام 'mp4v' أو 'XVID' لضمان التوافق مع FFmpeg
    out = cv2.VideoWriter(output_file.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # BoT-SORT Tracker for robust tracking
    tracker = "botsort.yaml"

    # استخدام st.spinner لإظهار مؤشر تحميل أثناء معالجة الفيديو
    with st.spinner("Analyzing video frames... This may take a few minutes."):
        results = model.track(
            source=video_path,
            conf=0.4,
            iou=0.5,
            tracker=tracker,
            persist=True,
            stream=True
        )

        # --------------------- ANALYSIS SETUP ---------------------
        color_ball = (0, 255, 255) # أصفر ساطع للكرة
        color_referee = (200, 200, 200) # رمادي للحكم
        color_possession = (0, 255, 0) # أخضر للاستحواذ

        last_owner_id = None
        possession_counter = defaultdict(int)
        # قائمة لتتبع حالات التمرير
        passes = []
        # تعيين ألوان الفريق المكتشفة لكل ID
        team_colors = {}
        # تجميع الإحصائيات حسب الفريق
        team_possession_counter = defaultdict(int)
        team_passes_counter = defaultdict(int)

        def get_average_color(frame, box):
            """يستخرج متوسط لون منطقة الاهتمام (ROI) الخاصة بالصندوق."""
            x1, y1, x2, y2 = box
            # التركيز على الجزء العلوي من الصندوق للحصول على لون القميص
            y_mid = y1 + int((y2 - y1) * 0.4)
            roi = frame[y1:y_mid, x1:x2]
            if roi.size == 0:
                return np.array([0, 0, 0])
            # استخدام np.mean لتجنب التحويل إلى RGB يدوياً
            return np.mean(roi.reshape(-1, 3), axis=0)

        def assign_team(player_id, color):
            """يُعين لون الفريق إلى اللاعب بناءً على أقرب مجموعة ألوان موجودة."""
            if player_id not in team_colors:
                # إذا كانت هذه هي المرة الأولى التي يظهر فيها اللاعب
                if len(team_colors) == 0:
                    # تعيين الفريق الأول إذا كان فارغاً
                    team_colors[player_id] = color
                else:
                    # محاولة تجميع اللون مع الألوان الموجودة
                    min_dist = 1e9
                    assigned_team_color = None
                    
                    # جمع الألوان الفريدة للفريق الحالي
                    unique_team_colors = list(set(tuple(c) for c in team_colors.values()))

                    for c in unique_team_colors:
                        dist = np.linalg.norm(color - c)
                        if dist < min_dist:
                            min_dist = dist
                            assigned_team_color = c

                    # إذا كان قريباً بدرجة كافية، عيّن لنفس لون الفريق
                    if min_dist < 40: # قيمة 40 هي حد تجريبي لتجميع الألوان
                        team_colors[player_id] = assigned_team_color
                    else:
                        # إذا كان بعيداً، فهذا لون فريق جديد (قد يكون فريق ثالث أو لون مختلف للحكم)
                        team_colors[player_id] = color
            return team_colors[player_id]

        progress_bar = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0

        # حلقة المعالجة
        for frame_data in results:
            frame = frame_data.orig_img.copy()
            processed_frames += 1
            progress_bar.progress(min(processed_frames / frame_count, 1.0))

            if frame_data.boxes.id is None:
                out.write(frame)
                continue

            # تحويل نتائج YOLOv8
            boxes = frame_data.boxes.xyxy.cpu().numpy()
            classes = frame_data.boxes.cls.cpu().numpy().astype(int)
            ids = frame_data.boxes.id.cpu().numpy().astype(int)

            balls, players = [], []

            for box, cls, track_id in zip(boxes, classes, ids):
                x1, y1, x2, y2 = map(int, box)

                if cls == 0: # الكرة
                    balls.append((track_id, (x1, y1, x2, y2)))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_ball, 2)
                    cv2.putText(frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color_ball, 2)

                elif cls in [1, 2]: # اللاعبون (افتراضياً)
                    # 1. تحديد لون الفريق
                    avg_color = get_average_color(frame, (x1, y1, x2, y2))
                    team_color_rgb = assign_team(track_id, avg_color)
                    
                    # 2. تعيين اسم ولون للعرض
                    # هذا الجزء من الشيفرة يفترض وجود فريقين فقط بناءً على متوسط السطوع
                    # ويفترض أن الفريق A داكن (أزرق) و الفريق B فاتح (أحمر)
                    if np.mean(team_color_rgb) < 128:
                        draw_color = (255, 0, 0) # أحمر للون الداكن (Team A)
                        team_name = "Team A"
                    else:
                        draw_color = (0, 0, 255) # أزرق للون الفاتح (Team B)
                        team_name = "Team B"

                    players.append((track_id, (x1, y1, x2, y2), team_name))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
                    cv2.putText(frame, f"{team_name} #{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, draw_color, 2)

                else: # الحكم أو أي شيء آخر
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_referee, 2)
                    cv2.putText(frame, "Referee", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, color_referee, 2)

            # Ball possession logic
            current_owner_id = None
            current_owner_team = None
            if len(balls) > 0 and len(players) > 0:
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
                
                # استخدام عتبة حيازة ثابتة (90)
                possession_threshold = 90
                
                if min_dist < possession_threshold:
                    possession_counter[current_owner_id] += 1
                    team_possession_counter[current_owner_team] += 1

                    # منطق التمرير
                    if last_owner_id is not None and current_owner_id != last_owner_id:
                        passes.append((last_owner_id, current_owner_id))
                        # تجميع التمريرات حسب الفريق الذي استلمها (أو قام بالتمريرة الأخيرة الناجحة)
                        team_passes_counter[current_owner_team] += 1
                    last_owner_id = current_owner_id
                else:
                    # فقدان الحيازة
                    last_owner_id = None


            # تمييز اللاعب المستحوذ على الكرة
            if current_owner_id is not None:
                for player_id, box, team_name in players:
                    if player_id == current_owner_id:
                        px1, py1, px2, py2 = box
                        cv2.rectangle(frame, (px1, py1), (px2, py2), color_possession, 4)
                        cv2.putText(frame, f"{team_name} #{player_id} HAS THE BALL",
                                    (px1, py1 - 15), cv2.FONT_HERSHEY_COMPLEX, 0.8, color_possession, 3)

            out.write(frame)

    cap.release()
    out.release()

    st.success("✅ Analysis completed successfully! See the results below.")

    # --------------------- RESULTS DISPLAY ---------------------
    st.header("📊 Analysis Summary")
    
    # تحويل عدادات الاستحواذ إلى نسب مئوية
    total_possession_frames = sum(team_possession_counter.values())
    
    if total_possession_frames > 0:
        team_a_possession = team_possession_counter["Team A"]
        team_b_possession = team_possession_counter["Team B"]
        
        team_a_percent = (team_a_possession / total_possession_frames) * 100
        team_b_percent = (team_b_possession / total_possession_frames) * 100
        
        st.subheader("Possession Breakdown")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Team A Possession", f"{team_a_percent:.1f}%")
        with col2:
            st.metric("Team B Possession", f"{team_b_percent:.1f}%")
            
        st.subheader("Passes Count")
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Team A Passes", team_passes_counter["Team A"])
        with col4:
            st.metric("Team B Passes", team_passes_counter["Team B"])
            
    else:
        st.info("No possession recorded. The video may be too short or the ball was not detected near players.")


    st.subheader("Processed Video")
    st.video(output_file.name)
    st.download_button("📥 Download Processed Video", data=open(output_file.name, "rb"), file_name="football_analysis.mp4")

    # تنظيف الملفات المؤقتة (خطوة مهمة)
    try:
        os.unlink(video_path)
        os.unlink(output_file.name)
    except Exception as e:
        print(f"Error cleaning up temporary files: {e}")


else:
    st.warning("Please upload a football video file (.mp4) to start the analysis.")
