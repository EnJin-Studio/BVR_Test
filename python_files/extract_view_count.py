import pandas as pd
import numpy as np

# Read CSV
df = pd.read_csv("video_meta.csv")

# Extract view_count and handle missing values
df["view_count"] = df["view_count"].fillna(0)

# Apply log transformation (add 1 to avoid log(0))
view_count_log = np.log1p(df["view_count"].astype(np.float32))  # log(1 + x)

# Save as .npy file
np.save("view_count_log.npy", view_count_log.values)
