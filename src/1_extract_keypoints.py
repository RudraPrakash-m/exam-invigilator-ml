import os
import cv2
import pandas as pd
from ultralytics import YOLO

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VIDEO_PATH = os.path.join(DATA_DIR, "videos", "train_video.mp4")
OUTPUT_CSV = os.path.join(DATA_DIR, "raw_keypoints.csv")
MODEL_PATH = os.path.join(BASE_DIR, "yolo11s-pose.pt")

os.makedirs(DATA_DIR, exist_ok=True)
# =============================================

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open video")

rows = []
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))

    # ✅ MUST USE TRACK
    results = model.track(frame, persist=True, conf=0.25, verbose=False)

    for r in results:
        if r.boxes is None or r.boxes.id is None or r.keypoints is None:
            continue

        ids = r.boxes.id.int().tolist()
        keypoints = r.keypoints.xyn.tolist()

        for i, tid in enumerate(ids):
            row = {
                "frame": frame_id,
                "track_id": tid
            }

            for j, kp in enumerate(keypoints[i]):
                row[f"x{j}"] = kp[0]
                row[f"y{j}"] = kp[1]

            rows.append(row)

    frame_id += 1

cap.release()

if not rows:
    raise RuntimeError("❌ No keypoints extracted — tracking failed")

pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
print("✅ Keypoints saved:", OUTPUT_CSV)
