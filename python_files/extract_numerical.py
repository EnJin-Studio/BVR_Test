import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Read CSV from csv_data/
df = pd.read_csv(os.path.join("csv_data", "video_meta.csv"))

# Columns to normalize
numeric_cols = ["duration", "pub_seconds_ago", "uploader_follower", "danmaku_count"]
df[numeric_cols] = df[numeric_cols].fillna(0)

# Apply log1p to selected long-tail columns
df["pub_seconds_ago"] = np.log1p(df["pub_seconds_ago"])
df["uploader_follower"] = np.log1p(df["uploader_follower"])
df["danmaku_count"] = np.log1p(df["danmaku_count"])
# duration remains unchanged

# Z-score normalization
scaler = StandardScaler()
numeric_features = scaler.fit_transform(df[numeric_cols].values.astype(np.float32))

# Save scaler parameters to npy_data/
np.save(os.path.join("npy_data", "scaler_mean.npy"), scaler.mean_)
np.save(os.path.join("npy_data", "scaler_std.npy"), scaler.scale_)

# Save normalized numeric features to npy_data/
np.save(os.path.join("npy_data", "numeric_features.npy"), numeric_features)
