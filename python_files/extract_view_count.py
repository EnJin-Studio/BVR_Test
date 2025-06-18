import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Read CSV
df = pd.read_csv(os.path.join("csv_data", "video_meta.csv"))

# Extract and log transform
df["view_count"] = df["view_count"].fillna(0)
view_log = np.log1p(df["view_count"].astype(np.float32))

# Z-score normalization
scaler = StandardScaler()
view_log_zscore = scaler.fit_transform(view_log.values.reshape(-1, 1)).squeeze()

# Save to npy_data/
np.save(os.path.join("npy_data", "view_count.npy"), view_log_zscore)
np.save(os.path.join("npy_data", "view_log_mean.npy"), scaler.mean_)
np.save(os.path.join("npy_data", "view_log_std.npy"), scaler.scale_)