import pandas as pd

df = pd.read_csv("window_features.csv")

df.loc[df["head_move"] > 0.008, "label"] = "Suspicious"
df.loc[df["shoulder_dist"] > 0.18, "label"] = "Suspicious"

df.to_csv("window_features.csv", index=False)

print(df["label"].value_counts())
