import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

CSV_PATH = os.path.join(DATA_DIR, "window_features.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "cheating_model.json")

os.makedirs(MODEL_DIR, exist_ok=True)
# =============================================

# Load data
df = pd.read_csv(CSV_PATH)

# Features & labels
X = df[["head_move", "shoulder_dist"]]
y = df["label"].map({"Normal": 0, "Suspicious": 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)

# Model
model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    eval_metric="logloss",
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# Save model
model.save_model(MODEL_PATH)
print("\n✅ Model saved to:", MODEL_PATH)
