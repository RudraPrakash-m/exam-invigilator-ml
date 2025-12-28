Here is a **clean, professional `README.md`** that accurately reflects **what you have built till now** — no exaggeration, no missing pieces.
This is **submission-ready** and **viva-ready**.

---

# 🎓 Smart Exam Invigilator System (AI-Based)

An AI-powered smart exam invigilation system that detects suspicious student behavior in real time using **pose estimation + machine learning + rule-based logic**.
The system is designed to reduce manual monitoring effort while minimizing false alarms caused by normal movements.

---

## 📌 Project Overview

Traditional exam invigilation relies heavily on human supervisors, which is error-prone and resource-intensive.
This project introduces an **automated invigilation system** that monitors students using a camera and identifies suspicious activities such as:

* Repeated head turning
* Looking sideways
* Hand movements towards nearby students
* Unusual body posture patterns

The system uses **YOLOv8 Pose Estimation** to extract keypoints and a **Machine Learning classifier** to analyze behavioral patterns over time.

---

## 🧠 Core Technologies Used

* **Python**
* **OpenCV** – video capture & visualization
* **YOLOv8 Pose (Ultralytics)** – human pose estimation
* **XGBoost** – machine learning classifier
* **NumPy / Pandas** – data processing
* **CSV logging & snapshot storage**

---

## 🏗️ System Architecture

```
Camera (Webcam / IP Camera)
        ↓
YOLOv8 Pose Estimation
        ↓
Keypoint Extraction (Head, Shoulders, Hands)
        ↓
Temporal Feature Engineering (30-frame window)
        ↓
ML Model + Rule-Based Logic
        ↓
Suspicious / Normal Classification
        ↓
Logging + Snapshot Capture
```

---

## 📂 Project Folder Structure

```
exam_invigilator_1/
│
├── src/
│   ├── 1_extract_keypoints.py
│   ├── 2_feature_engineering.py
│   ├── 3_train_model.py
│   └── 4_live_detection.py   ← (current stable version)
│
├── data/
│   ├── videos/
│   ├── raw_keypoints.csv
│   └── window_features.csv
│
├── models/
│   └── cheating_model.json
│
├── logs/
│   └── events.csv
│
├── snapshots/
│   └── *.jpg
│
├── yolo11s-pose.pt
└── README.md
```

---

## ⚙️ How the System Works

### 1️⃣ Pose Detection

* YOLOv8 Pose model detects human body keypoints in each frame.
* Keypoints include head, shoulders, wrists, etc.

### 2️⃣ Temporal Analysis

* Keypoints are stored in a **sliding window of 30 frames** (~1 second).
* This avoids reacting to single-frame noise.

### 3️⃣ Feature Extraction

Features currently used:

* Head movement magnitude (with noise threshold)
* Shoulder distance
* Left & right wrist movement

### 4️⃣ Hybrid Decision Logic

* **ML Model (XGBoost)** predicts suspicious probability.
* **Rule-based overrides** detect clear hand movements.
* Small natural head movements are ignored using thresholds.

### 5️⃣ Output

* Bounding box + label (`Normal` / `Suspicious`)
* Event logged to CSV
* Snapshot captured for evidence

---

## ✅ Key Improvements Implemented

✔ Reduced false positives from natural head movement
✔ Added hand-movement-based cheating detection
✔ Used motion persistence instead of single-frame triggers
✔ Hybrid ML + rule-based logic (industry practice)
✔ Stable real-time performance

---

## 📊 Example Log Entry

```
timestamp, track_id, probability, severity, label
2025-01-04 11:32:15, 2, 0.91, , Suspicious
```

---

## 🧪 How to Run the Project

### Install Dependencies

```bash
pip install ultralytics opencv-python xgboost numpy pandas
```

### Run Live Detection

```bash
python src/4_live_detection.py
```

Press **`q`** to exit.

---

## 📷 Camera Support

* ✔ Laptop Webcam
* ✔ External USB Camera
* ✔ IP Webcam (Android phone) *(commented in code)*

---

## 🚧 Current Limitations

* No audio alerts (planned via ESP32)
* Does not yet detect student-to-student eye contact
* Works best for seated exam environments

---

## 🔮 Future Enhancements

* ESP32-based buzzer / speaker alerts
* Student interaction detection
* Eye gaze estimation
* Strike count per student
* Dashboard for supervisors
* Cloud-based logging

---

## 🎓 Academic Relevance

This project demonstrates:

* Computer Vision
* Temporal Machine Learning
* Hybrid AI system design
* Real-world false positive handling
* Practical surveillance system design

---

## 📢 Final Note

This system is designed as an **assistive tool**, not a replacement for human invigilators.
All detections should be reviewed by a supervisor.

---

If you want next:

* **Architecture diagram (for PPT)**
* **Viva questions & answers**
* **Evaluation metrics section**
* **ESP32 integration README**

Just say the word.
