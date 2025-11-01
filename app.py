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

# تعريف أسماء الفئات بناءً على تدريب النموذج
CLASS_NAMES = {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}

# مسار النموذج المُدرّب (يجب التأكد من وجوده)
MODEL_PATH = "yolov8m-football_ball_only.pt" 

# الألوان الثابتة للعرض (BGR)
COLOR_BALL = (0, 255, 255) # أصفر/سماوي
COLOR_REFEREE_DISPLAY = (0, 165, 255) # برتقالي/عنبري
COLOR_GOALKEEPER_DISPLAY = (255, 255, 0) # أصفر فاتح للحارس

# --- ألوان العرض الجديدة للوضوح (BGR) ---
DISPLAY_COLOR_A = (0, 0, 255) # أحمر للفريق A
DISPLAY_COLOR_B = (255, 0, 0) # أزرق للفريق B
# ---------------------------------------

# ثوابت التعلم التلقائي
AUTO_LEARNING_FRAMES = 150 # زيادة عينات التعلم لـ 150 إطارًا
# زيادة التسامح لتحسين الفصل اللوني، أو استخدام منطق المسافة النسبية
# نستخدم قيمة أعلى هنا لتقليل حالات "Unassigned" غير الضرورية
COLOR_TOLERANCE = 120 

# --- UTILITY FOR COLOR ANALYSIS (K-Means Clustering - Pure NumPy) ---

def get_average_color(frame, box):
    """
    يستخلص متوسط لون البيكسلات في الثلث العلوي من صندوق التحديد (القميص) كـ BGR.
    """
    x1, y1, x2, y2 = map(int, box)
    # التركيز على الثلث العلوي كمنطقة القميص لتقليل تأثير العشب
    roi = frame[y1: int(y1 + (y2 - y1) / 3), x1:x2]
    if roi.size == 0:
        # يُرجع مصفوفة NumPy صفرية متوقعة (float32)
        return np.array([0., 0., 0.], dtype=np.float32)
    # حساب متوسط اللون في منطقة القميص وإرجاعه كـ float32
    return np.mean(roi.reshape(-1,3), axis=0).astype(np.float32)

# Function to perform simplified K-Means clustering using NumPy
def simple_kmeans_numpy(data, k=2, max_iters=10):
    """تنفيذ بسيط لخوارزمية K-Means باستخدام NumPy."""
    
    # يجب أن تكون البيانات من النوع float32
    data = data.astype(np.float32) 
    
    if data.shape[0] < k:
        return None

    # 1. تهيئة k مراكز عشوائياً
    indices = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[indices]

    for _ in range(max_iters):
        # 2. خطوة التعيين: إيجاد أقرب مركز لكل نقطة بيانات
        # حساب المسافة الإقليدية لجميع النقاط في خطوة واحدة
        distances = np.sqrt(np.sum((data - centroids[:, np.newaxis])**2, axis=2))
        labels = np.argmin(distances, axis=0)

        # 3. خطوة التحديث: إعادة حساب المراكز
        new_centroids = np.array([data[labels == i].mean(axis=0) 
                                 if np.any(labels == i) else centroids[i] 
                                 for i in range(k)])
        
        # التحقق من التقارب
        if np.allclose(centroids, new_centroids):
            break
        
        centroids = new_centroids
        
    return centroids.astype(np.float32)


# قاموس لتخزين تعيين الفريق (Team A/Team B) لـ ID اللاعب بشكل ثابت
team_assignment_map = {} 
TEAM_A_CENTER = None # سيتم تخزينها كـ np.ndarray (float32)
TEAM_B_CENTER = None # سيتم تخزينها كـ np.ndarray (float32)


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

    # التأكد من أن color_np هي float32 لتماشيها مع مراكز الألوان
    color_np = color.astype(np.float32)
    
    # حساب المسافة الإقليدية (مسافة الألوان). كلا المركزين مصفوفات NumPy
    dist_a = np.linalg.norm(color_np - TEAM_A_CENTER)
    dist_b = np.linalg.norm(color_np - TEAM_B_CENTER)
    
    assigned_team_name = "Unassigned"
    
    # 2. التصنيف: استخدم الأقرب فقط، واستخدم التسامح لفحص ما إذا كان قريباً بما فيه الكفاية
    if dist_a < dist_b and dist_a < COLOR_TOLERANCE:
        assigned_team_name = "Team A"
    elif dist_b < dist_a and dist_b < COLOR_TOLERANCE:
        assigned_team_name = "Team B"

    # حفظ التعيين إذا نجح
    if assigned_team_name != "Unassigned":
        team_assignment_map[player_id] = assigned_team_name
        
    return assigned_team_name


def determine_team_colors(kit_colors):
    """
    يطبق خوارزمية K-Means لتحديد مركزي اللون (K=2) للفريقين.
    """
    global TEAM_A_CENTER, TEAM_B_CENTER
    
    # يجب أن تكون العينات كافية
    if len(kit_colors) < 50: 
        return 
    
    # تحويل القائمة إلى مصفوفة NumPy من النوع float32
    colors_np = np.array(kit_colors, dtype=np.float32)
    
    # تطبيق K-Means
    centers = simple_kmeans_numpy(colors_np, k=2)
    
    if centers is None or centers.shape[0] < 2:
        return 
    
    # تحديد الفريق A و B بناءً على اللمعان (الفريق A هو الداكن/الأقل لمعاناً)
    # اللمعان يُحسب هنا كمتوسط لقيم BGR (بالنظر لكون BGR قيم موجبة)
    luminosity_A_center = np.mean(centers[0])
    luminosity_B_center = np.mean(centers[1])
    
    # الفريق A هو صاحب اللمعان الأقل (الداكن)
    if luminosity_A_center < luminosity_B_center:
        # التصحيح: يتم تخزين المراكز كـ np.ndarray (float32) وليس قوائم int
        TEAM_A_CENTER = centers[0]
        TEAM_B_CENTER = centers[1]
    else:
        TEAM_A_CENTER = centers[1]
        TEAM_B_CENTER = centers[0]


# --- 2. CORE PROCESSING LOGIC ---

@st.cache_resource
def load_model():
    """يحمّل نموذج YOLO لمرة واحدة ويخزنه مؤقتاً."""
    try:
        st.info(f"جارٍ محاولة تحميل نموذج YOLO: {MODEL_PATH}")
        # تفضيل استخدام وضع التتبع لتحديد الـ ID
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحميل نموذج YOLO. Error: {e}")
        st.stop()

def process_video(uploaded_video_file, model):
    
    # إعادة تعيين المتغيرات العالمية عند بدء تشغيل جديد
    global team_assignment_map, TEAM_A_CENTER, TEAM_B_CENTER
    team_assignment_map = {} 
    TEAM_A_CENTER = None
    TEAM_B_CENTER = None
    
    kit_colors_for_learning = [] # قائمة لتخزين الألوان للإطارات الأولى

    # حفظ الفيديو المُحمّل في ملف مؤقت
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_video_file.read())
        video_path = tfile.name

    output_video_path = os.path.join(tempfile.gettempdir(), "football_tracking_output.mp4")

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # استخدام 'mp4v' للتوافق مع H.264
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    # --- إعدادات التتبع المُحسَّنة ---
    results = model.track(
        source=video_path,
        conf=0.40,  # ثقة الكشف
        iou=0.7,    # تداخل الاتحاد على الاكتشاف
        persist=True,
        tracker="botsort.yaml", 
        stream=True,
        verbose=False
    )
    # -------------------------------------------------------------------

    frame_num = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0, text="يتم تحليل الإطارات الأولية لتحديد ألوان الفرق...")
    
    # قائمة لتخزين الألوان المحددة (لغرض العرض في النهاية)
    final_centers_display = {"Team A": None, "Team B": None} 

    for frame_data in results:
        frame_num += 1
        frame = frame_data.orig_img.copy()

        # تحديث شريط التقدم
        if total_frames > 0:
            progress_bar.progress(min(frame_num / total_frames, 1.0), 
                                  text="جاري معالجة الإطارات...")

        # --- محاولة استخلاص الصناديق والـ ID ---
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
        
        # 1. مرحلة التعلم التلقائي
        if frame_num <= AUTO_LEARNING_FRAMES:
            if boxes is not None:
                for box, cls, track_id in zip(boxes, classes, ids):
                    # تجاهل الكرة والحكم (cls 0 و 3)
                    if cls in [1, 2]: # لاعب أو حارس
                        avg_bgr_color = get_average_color(frame, box)
                        kit_colors_for_learning.append(avg_bgr_color)
            
            # إذا وصلنا لنهاية مرحلة التعلم، قم بالحساب
            if frame_num == AUTO_LEARNING_FRAMES:
                 if len(kit_colors_for_learning) >= 50: # التأكد من جمع عينات كافية
                     determine_team_colors(kit_colors_for_learning)
                     
                     if TEAM_A_CENTER is not None:
                        # تحويل مراكز الألوان إلى قائمة أعداد صحيحة للعرض فقط
                        final_centers_display["Team A"] = TEAM_A_CENTER.astype(int).tolist()
                        final_centers_display["Team B"] = TEAM_B_CENTER.astype(int).tolist()
                        
                        progress_bar.progress(min(AUTO_LEARNING_FRAMES / total_frames, 1.0), 
                                              text="تم تحديد مراكز الألوان بنجاح. بدء التتبع...")
                     else:
                         st.warning("تعذر تحديد مراكز ألوان واضحة باستخدام K-Means. سيتم استخدام ألوان احتياطية.")
                         # fallback: يجب أن تكون مصفوفة NumPy
                         TEAM_A_CENTER = np.array([0, 0, 255], dtype=np.float32) 
                         TEAM_B_CENTER = np.array([255, 0, 0], dtype=np.float32)
                         final_centers_display["Team A"] = TEAM_A_CENTER.astype(int).tolist()
                         final_centers_display["Team B"] = TEAM_B_CENTER.astype(int).tolist()
                 else:
                    st.warning("لم يتم الكشف عن عدد كافٍ من الألوان المميزة في الإطارات الأولية. سيتم استخدام ألوان احتياطية.")
                    # fallback: يجب أن تكون مصفوفة NumPy
                    TEAM_A_CENTER = np.array([0, 0, 255], dtype=np.float32) 
                    TEAM_B_CENTER = np.array([255, 0, 0], dtype=np.float32)
                    final_centers_display["Team A"] = TEAM_A_CENTER.astype(int).tolist()
                    final_centers_display["Team B"] = TEAM_B_CENTER.astype(int).tolist()

        # 2. مرحلة التتبع والتصنيف (بعد تحديد المراكز)
        if TEAM_A_CENTER is not None and boxes is not None:
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
                    
                    # 1. تحديد لون القميص (يُرجع np.float32 array)
                    avg_bgr_color = get_average_color(frame, (x1, y1, x2, y2))
                    
                    # 2. التعيين للفريق بناءً على المراكز اللونية المستخلصة (باستخدام مصفوفات float)
                    assigned_team_name = assign_team_by_reference(
                        track_id_int, avg_bgr_color
                    )
                    
                    team_label = assigned_team_name

                    # 3. تحديد لون العرض بناءً على التعيين (استخدام الألوان الواضحة)
                    if team_label == "Team A":
                        color = DISPLAY_COLOR_A # استخدام الأحمر للعرض
                    elif team_label == "Team B":
                        color = DISPLAY_COLOR_B # استخدام الأزرق للعرض
                    else:
                        color = (255, 255, 255) # اللاعبون غير المصنفين أبيض

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
        
        # كتابة الإطار إلى ملف الإخراج
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
    st.markdown('<div class="main-title">⚽ كشف وتتبع لاعبي كرة القدم آلياً </div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Layout for inputs and preview
    col1, col2 = st.columns([1, 1])
    
    # يجب إعلان المتغيرات العالمية مرة أخرى هنا للوصول إلى قيمها النهائية بعد المعالجة
    global TEAM_A_CENTER, TEAM_B_CENTER
    
    uploaded_file = None
    with col1:
        st.subheader("1. تحميل الفيديو (كشف الألوان تلقائي) 🎨")
        uploaded_file = st.file_uploader("تحميل فيديو MP4 لمباراة كرة قدم", type=["mp4"])

        st.markdown("---")
        st.markdown(f"""
            **منطق تعيين الفرق (تلقائي بالكامل):** يحلل النظام أول {AUTO_LEARNING_FRAMES} إطارًا لتحديد لوني القميص الرئيسيين تلقائيًا باستخدام خوارزمية K-Means.
            - **الفريق A (الأطقم الداكنة):** يُعرض باللون الأحمر ({DISPLAY_COLOR_A} BGR).
            - **الفريق B (الأطقم الفاتحة):** يُعرض باللون الأزرق ({DISPLAY_COLOR_B} BGR).
            
            *لا يتطلب إدخال لوني يدوي.*
        """)
        
        st.markdown(f"**لون عرض الحارس (ثابت):** {COLOR_GOALKEEPER_DISPLAY}")
        st.markdown(f"**لون عرض الحكم (ثابت):** {COLOR_REFEREE_DISPLAY}")
        st.markdown("---")


    # Pre-Analysis Video Preview
    with col2:
        if uploaded_file is not None:
            st.subheader("2. معاينة الفيديو الأصلي")
            st.video(uploaded_file)
            st.success("تم تحميل الفيديو بنجاح!")
        else:
            st.info("الرجاء تحميل فيديو لتمكين التحليل.")


    st.markdown("---")

    # Processing Button
    if uploaded_file is not None:
        if st.button("بدء التتبع والتعيين التلقائي للفرق", key="start_analysis", type="primary"):
            try:
                # Execution
                output_video_path, detected_centers = process_video(uploaded_file, model)

                st.success("اكتمل التتبع والتصنيف بنجاح! 🎉")
                st.markdown("---")

                # --- Output Section ---
                st.subheader("3. مخرجات الفيديو المُعالج")
                
                # Display the determined colors (optional, for feedback)
                st.markdown(f"""
                    #### مراكز الألوان المكتشفة (BGR - المُستخدمة في منطق التصنيف)
                    - **الفريق A (مركز الطقم الداكن):** `{detected_centers['Team A']}`
                    - **الفريق B (مركز الطقم الفاتح):** `{detected_centers['Team B']}`
                """)
                
                # Video Display
                with open(output_video_path, 'rb') as f:
                    output_video_bytes = f.read()
                st.video(output_video_bytes)

                # Download button for Video
                st.download_button(
                    label="تحميل الفيديو المُعالج (MP4)",
                    data=output_video_bytes,
                    file_name="football_tracking_output.mp4",
                    mime="video/mp4",
                    type="secondary"
                )

            except Exception as e:
                st.error("حدث خطأ أثناء معالجة الفيديو.")
                st.exception(e)

    elif uploaded_file is None:
        st.info("قم بتحميل ملف فيديو لتمكين التحليل.")

if __name__ == '__main__':
    streamlit_app()
