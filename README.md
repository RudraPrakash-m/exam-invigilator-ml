# 🎓 Smart Exam Invigilator System

**Pose-Based AI Cheating Detection**

---

## 📌 Project Description

The **Smart Exam Invigilator System** is an AI-based surveillance solution designed to monitor examination halls and detect **suspicious student behavior** in real time using **pose estimation and machine learning**.

The system analyzes **head movement, hand movement, and body posture** of students over time and classifies behavior as **Normal** or **Suspicious** without using face recognition, ensuring privacy.

---

## 🧠 Core Idea

Instead of detecting faces or identities, the system:

* Detects **people**
* Extracts **pose keypoints**
* Tracks **motion patterns across frames**
* Classifies behavior using a trained ML model
* Assigns each student to a **fixed seat zone (A1, A2, …)**

---

## 🧩 Technologies Used

| Component            | Technology                |
| -------------------- | ------------------------- |
| Video Processing     | OpenCV                    |
| Pose Detection       | YOLOv8 Pose               |
| Machine Learning     | XGBoost                   |
| Programming Language | Python                    |
| Data Storage         | CSV                       |
| Camera Support       | Laptop Webcam / MP4 Video |

---

## 📁 Project Structure (From ZIP)

```
exam_invigilator_1/
│
├── src/
│   ├── 1_extract_keypoints.py     # Extract pose keypoints from video
│   ├── 2_build_features.py        # Build temporal features
│   ├── 3_train_model.py           # Train ML model
│   └── 4_live_detection.py        # Real-time detection (webcam / video)
│
├── data/
│   ├── videos/
│   │   └── train_video.mp4        # Training / testing video
│   └── window_features.csv        # Extracted feature dataset
│
├── models/
│   └── cheating_model.json        # Trained XGBoost model
│
├── logs/
│   └── events.csv                 # Detection logs
│
├── snapshots/
│   └── *.jpg                      # Evidence snapshots
│
├── yolo11s-pose.pt                # YOLO Pose model
├── requirements.txt
└── README.md
```

---

## 🔄 Complete Pipeline

```
MP4 / Camera Input
        ↓
YOLO Pose Detection
        ↓
Pose Keypoints (17 body points)
        ↓
Temporal Feature Extraction (Windowed)
        ↓
XGBoost Classifier
        ↓
Suspicious / Normal Decision
        ↓
Zone-Based Label (A1, A2)
        ↓
Logging + Snapshots
```

---

## 🧍 Pose Keypoints Used

The system uses YOLO’s **COCO 17-keypoint format**:

| Feature          | Keypoints        |
| ---------------- | ---------------- |
| Head movement    | Nose (0)         |
| Hand movement    | Wrists (9, 10)   |
| Body orientation | Shoulders (5, 6) |

These keypoints are analyzed over multiple frames to detect meaningful behavior.

---

## 🪟 Sliding Window & Cooldown

* **Sliding Window (30 frames)**
  Ensures decisions are based on motion over time, not single frames.

* **Cooldown Mechanism**
  Prevents repeated alerts/logs for the same student within a short time window.

This keeps the system **stable and realistic**.

---

## 🪑 Zone-Based Identification

Each student is assigned a **seat zone**:

```
A1   A2
```

### Why zone-based IDs?

* Exam seating is fixed
* No tracker ID flickering
* Easy for invigilators to understand
* No personal identity stored

Displayed labels:

```
A1
Suspicious A2
```

---

## 📊 Output & Evidence

### On Screen

* 🟢 Green box → Normal
* 🔴 Red box → Suspicious
* Label → Zone ID

### Logs (`logs/events.csv`)

```
timestamp, zone, probability, label
```

### Snapshots

* Automatically captured when suspicious activity is detected
* Stored for later review

---

## 🎥 Running the Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Step-by-Step Execution

#### Step 1: Extract Keypoints

```bash
python src/1_extract_keypoints.py
```

#### Step 2: Build Features

```bash
python src/2_build_features.py
```

#### Step 3: Train Model

```bash
python src/3_train_model.py
```

#### Step 4: Run Detection (Webcam or Video)

```bash
python src/4_live_detection.py
```

---

## 🎥 Input Modes Supported

### ✔ MP4 Video (Testing)

```python
cap = cv2.VideoCapture("data/videos/train_video.mp4")
```

Used for:

* Training
* Debugging
* Evaluation

### ✔ Laptop Webcam (Live)

```python
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
```

Used for:

* Real-time monitoring
* Live demo

---

## 🔐 Privacy & Ethics

* ❌ No face recognition
* ❌ No identity storage
* ✔ Pose-only analysis
* ✔ GDPR-friendly approach

---

## 🚀 Future Scope

* Robot-based invigilator (ESP32)
* Multi-camera fusion
* Audio alerts
* Dashboard monitoring
* Depth-aware detection

---

## 🎓 Academic Relevance

This project demonstrates:

* Computer Vision
* Pose Estimation
* Temporal Machine Learning
* Real-world system design
* Ethical AI implementation

---

## 👨‍💻 Author

**Rudra**
B.Tech – Computer Science Engineering
AI & Smart Surveillance Systems

---

## ✅ Final Note

This project is designed to be **realistic, explainable, and deployable**, not just a demo.
It closely follows how **real AI surveillance systems are engineered**.
