import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Read CSV
df = pd.read_csv("video_meta.csv")

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

# ✅ 保存 scaler 参数
np.save("scaler_mean.npy", scaler.mean_)
np.save("scaler_std.npy", scaler.scale_)

# Save as .npy file
np.save("numeric_features.npy", numeric_features)
