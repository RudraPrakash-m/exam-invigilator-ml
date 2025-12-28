import os
import pandas as pd
import numpy as np

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT_CSV = os.path.join(DATA_DIR, "raw_keypoints.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "window_features.csv")
# =============================================

WINDOW = 30  # frames (~1 second at 30 FPS)

df = pd.read_csv(INPUT_CSV)

features = []

for tid in df["track_id"].unique():
    person = df[df["track_id"] == tid].reset_index(drop=True)

    for i in range(0, len(person) - WINDOW, WINDOW):
        win = person.iloc[i:i + WINDOW]

        # Head (nose) movement
        head_x = win["x0"]
        head_y = win["y0"]

        head_movement = np.mean(
            np.sqrt(np.diff(head_x) ** 2 + np.diff(head_y) ** 2)
        )

        # Shoulder width change (posture variation)
        shoulder_dist = np.mean(
            np.abs(win["x5"] - win["x6"])
        )

        features.append({
            "track_id": tid,
            "head_move": head_movement,
            "shoulder_dist": shoulder_dist,
            "label": "Normal"  # TEMPORARY
        })

pd.DataFrame(features).to_csv(OUTPUT_CSV, index=False)
print("✅ window_features.csv created")
