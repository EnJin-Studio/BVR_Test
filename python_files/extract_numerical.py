import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Read CSV
df = pd.read_csv("video_meta.csv")

# Ensure these columns exist and handle missing values
numeric_cols = ["duration", "pub_seconds_ago", "uploader_follower", "danmaku_count"]
df[numeric_cols] = df[numeric_cols].fillna(0)

# Z-score normalization
scaler = StandardScaler()
numeric_features = scaler.fit_transform(df[numeric_cols].values.astype(np.float32))

# Save as .npy file
np.save("numeric_features.npy", numeric_features)
