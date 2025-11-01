# ⚽ Deep Learning Project: Football Player and Ball Detection & Tracking with YOLOv8

## 🧠 Project Overview
This project leverages **Deep Learning (DL)** and **Computer Vision** techniques to automatically analyze football match videos.  
It uses **fine-tuned Ultralytics YOLOv8 models** to detect and persistently track players, the ball, goalkeepers, and referees.  
The main goal is to provide automated **team classification**, **ball possession tracking**, and **match statistics** generation.

---

## 🌟 Key Features

### 1. Accurate Detection and Multi-Object Tracking (MOT)
- **Object Detection:**  
  Utilizes customized YOLOv8 models (`YOLOv8m`, `YOLOv8l`) trained on football datasets to detect:
  - Player  
  - Ball  
  - Goalkeeper  
  - Referee  

- **Persistent Tracking:**  
  Implements advanced MOT algorithms (**BotSORT** and **ByteTrack**) to maintain stable IDs for detected objects across frames.

---

### 2. Team Classification and Ball Possession Analysis
- **Automatic Team Assignment:**  
  Players are automatically classified into **Team A (Darker kit)** and **Team B (Lighter kit)**.  
  The system identifies the two primary team colors using **K-Means Clustering** on player jersey regions from the initial frames.

- **Possession Logic:**  
  Determines the player in possession of the ball using **Euclidean distance** between the ball’s center and each player’s foot position.  
  A proximity threshold (e.g., `180 pixels`) ensures stable possession assignment.

- **Match Statistics (In-Video Display & Logging):**
  - Tracks the number of frames each player and team has possession.  
  - Counts total passes between different player IDs.  
  - Logs detailed possession data into a CSV file.

---

### 3. Interactive Web Application (Streamlit)
A **user-friendly Streamlit app** provides real-time visualization and interaction:
- Upload your football match video (`.mp4`).  
- Watch real-time detection and tracking results.  
- View team color assignments and ball possession highlights.  

---

## 🛠 Technologies and Tools

| Tool / Library | Purpose |
|-----------------|----------|
| **Python** | Main programming language |
| **Ultralytics YOLOv8** | Core DL model for detection & tracking |
| **OpenCV (cv2)** | Video frame processing and visualization |
| **BotSORT / ByteTrack** | Multi-object tracking algorithms |
| **Streamlit** | Web-based interface for visualization |
| **NumPy & Pandas** | Data manipulation and CSV logging |
| **Google Colab / Kaggle** | Training and testing environments |

---

## 🚀 Setup and Installation

### 🔹 Prerequisites
- Python **3.x**
- GPU acceleration (recommended for performance)
- YOLOv8-compatible environment

### 🔹 Installation Steps
```bash
# 1️⃣ Install Ultralytics YOLOv8
pip install ultralytics==8.3.15

# 2️⃣ Install additional dependencies
pip install opencv-python numpy pandas streamlit
```

### 🔹 Model Files
Make sure the trained YOLO model weights are placed correctly:  
Example:
```
yolov8m-football_ball_only.pt
```
This file should be in the same directory as your `app.py` or notebook.

---

## ▶️ Running the Application

### 🖥 Streamlit Web App
Run the app locally:
```bash
streamlit run app.py
```

> 💡 You can rename or adjust the script name if needed (e.g., `app (1).py`).

---

### 📓 Jupyter / Colab Notebook
You can also run the detection and tracking pipeline inside a notebook:
```python
from ultralytics import YOLO

model = YOLO("path/to/model.pt")

results = model.track(
    source="/content/videos/video.mp4",
    conf=0.4,
    iou=0.5,
    tracker="bytetrack.yaml",
    stream=True
)
```

---

## 📊 Example Output
- Real-time bounding boxes with player, ball, and referee labels.  
- Team A and Team B differentiated by color overlay.  
- Ball possession dynamically highlighted.  
- CSV file (`possession_log.csv`) generated with detailed match stats.

---

## 📁 Project Structure (Example)
```
📂 Football-Tracking-YOLOv8
│
├── app.py
├── yolov8m-football_ball_only.pt
├── requirements.txt
├── sample_video.mp4
├── possession_log.csv
├── README.md
└── /notebooks
     └── tracking_demo.ipynb
```

---

## 📢 Acknowledgements
- **Ultralytics YOLOv8** – for state-of-the-art object detection models  
- **ByteTrack** and **BotSORT** – for efficient multi-object tracking  
- **OpenCV & Streamlit** – for real-time visualization and deployment  

---

### 🧑‍💻 Author
**Osama Ashraf Eid**  
🎓 Faculty of Computers and Information, Fayoum University – AI Department  
📧 [osama.os.gh.2004.2003@gmail.com](mailto:osama.os.gh.2004.2003@gmail.com)  
📍 Egypt

---

⭐ **If you like this project, don’t forget to star the repo on GitHub!**
