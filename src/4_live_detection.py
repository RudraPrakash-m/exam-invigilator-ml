import os
import cv2
import csv
import time
import heapq
import numpy as np
import xgboost as xgb
from datetime import datetime
from ultralytics import YOLO

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "cheating_model.json")
POSE_MODEL_PATH = os.path.join(BASE_DIR, "yolo11s-pose.pt")

LOG_DIR = os.path.join(BASE_DIR, "logs")
SNAP_DIR = os.path.join(BASE_DIR, "snapshots")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SNAP_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "events.csv")

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(
            ["timestamp", "track_id", "zone", "probability", "severity", "label"]
        )

# ================= MODELS =================
clf = xgb.XGBClassifier()
clf.load_model(MODEL_PATH)

pose = YOLO(POSE_MODEL_PATH)

# ================= CAMERA =================
cap = cv2.VideoCapture(1, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
cap.set(cv2.CAP_PROP_FPS, 30)


# ================= GRID =================
ROWS = 2
COLS = 2

# ================= PARAMETERS =================
WINDOW = 45
SNAP_COOLDOWN = 1
ZONE_COOLDOWN = 15
ROBOT_COOLDOWN = 6

buffers = {}
last_snapshot_time = {}
zone_in_queue = set()
zone_last_handled = {}
event_queue = []

last_robot_action = 0

# ================= ZONE FUNCTION =================
def get_zone(x1, y1, x2, y2, w, h):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    col = min(int(cx / (w / COLS)), COLS - 1)
    row = min(int(cy / (h / ROWS)), ROWS - 1)

    return f"{chr(ord('A') + row)}{col + 1}"

# ================= WINDOW =================
cv2.namedWindow("Smart Exam Invigilator", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Smart Exam Invigilator", 1280, 720)

# ================= MAIN LOOP =================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (960, 540))
    h, w, _ = frame.shape

    results = pose.track(frame, persist=True, conf=0.15, verbose=False)

    for r in results:
        if r.boxes is None or r.boxes.id is None or r.keypoints is None:
            continue

        ids = r.boxes.id.int().tolist()
        boxes = r.boxes.xyxy.int().tolist()
        kps = r.keypoints.xyn.tolist()

        for i, tid in enumerate(ids):
            buffers.setdefault(tid, []).append(kps[i])
            if len(buffers[tid]) < WINDOW:
                continue

            arr = np.array(buffers[tid])

            x1, y1, x2, y2 = boxes[i]
            box_area = (x2 - x1) * (y2 - y1)

            # ❌ Ignore extremely small skeletons
            if box_area < 8000:
                buffers[tid].pop(0)
                continue

            # ===== ADAPTIVE THRESHOLDS =====
            if box_area < 15000:      # far student
                head_thresh = 0.015
                hand_thresh = 0.04
            else:                     # near student
                head_thresh = 0.03
                hand_thresh = 0.07

            # ===== FEATURE EXTRACTION =====
            head_deltas = np.sqrt(
                np.diff(arr[:, 0, 0])**2 +
                np.diff(arr[:, 0, 1])**2
            )
            head_move = np.mean(head_deltas[head_deltas > head_thresh]) if np.any(head_deltas > head_thresh) else 0

            left_hand = np.mean(np.sqrt(
                np.diff(arr[:, 9, 0])**2 +
                np.diff(arr[:, 9, 1])**2
            ))
            right_hand = np.mean(np.sqrt(
                np.diff(arr[:, 10, 0])**2 +
                np.diff(arr[:, 10, 1])**2
            ))

            shoulder_dist = np.mean(np.abs(arr[:, 5, 0] - arr[:, 6, 0]))

            X = np.array([[head_move, shoulder_dist]])
            prob = clf.predict_proba(X)[0][1]

            if left_hand > hand_thresh or right_hand > hand_thresh:
                prob = max(prob, 0.9)

            label = "Suspicious" if prob > 0.7 else "Normal"
            color = (0, 0, 255) if label == "Suspicious" else (0, 255, 0)

            zone = get_zone(x1, y1, x2, y2, w, h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {zone}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # ===== EVENT HANDLING =====
            if label == "Suspicious":
                now = time.time()

                if zone in zone_last_handled and now - zone_last_handled[zone] < ZONE_COOLDOWN:
                    buffers[tid].pop(0)
                    continue

                severity = (
                    "mild" if prob < 0.75 else
                    "medium" if prob < 0.88 else
                    "severe"
                )
                priority = {"severe": 0, "medium": 1, "mild": 2}[severity]

                if zone not in zone_in_queue:
                    heapq.heappush(event_queue, (priority, now, tid, zone, prob, severity))
                    zone_in_queue.add(zone)

                if tid not in last_snapshot_time or now - last_snapshot_time[tid] > SNAP_COOLDOWN:
                    last_snapshot_time[tid] = now
                    cv2.imwrite(os.path.join(SNAP_DIR, f"{zone}_{tid}_{int(now)}.jpg"), frame)

                    with open(LOG_FILE, "a", newline="") as f:
                        csv.writer(f).writerow(
                            [datetime.now(), tid, zone, round(prob, 3), severity, label]
                        )

            buffers[tid].pop(0)

    # ===== ROBOT DISPATCH =====
    now = time.time()
    if event_queue and (now - last_robot_action) > ROBOT_COOLDOWN:
        last_robot_action = now
        priority, ts, tid, zone, prob, severity = heapq.heappop(event_queue)
        zone_in_queue.discard(zone)
        zone_last_handled[zone] = time.time()

        print(f"[ROBOT TASK] Go to zone {zone} | Severity: {severity}")

    cv2.imshow("Smart Exam Invigilator", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()