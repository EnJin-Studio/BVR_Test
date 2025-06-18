import numpy as np
import pandas as pd
import os

# Load the CSV
df = pd.read_csv(os.path.join("csv_data", "video_meta.csv"))
bvids_csv = df["bvid"].astype(str).tolist()

# Load image features and their corresponding bvid order from npy_data folder
clip_feats = np.load(os.path.join("npy_data", "picture_features.npy"))   # shape: [M, 512]
clip_bvids = np.load(os.path.join("npy_data", "picture_bvids.npy"))      # shape: [M]

# Build a mapping from bvid to index
bvid_to_index = {bvid: idx for idx, bvid in enumerate(clip_bvids)}

# Create aligned array
aligned_clip_feats = np.zeros((len(bvids_csv), clip_feats.shape[1]), dtype=np.float32)

missing = 0
for i, bvid in enumerate(bvids_csv):
    if bvid in bvid_to_index:
        aligned_clip_feats[i] = clip_feats[bvid_to_index[bvid]]
    else:
        aligned_clip_feats[i] = np.zeros(clip_feats.shape[1])
        missing += 1

print(f"[INFO] Alignment complete. Missing image features: {missing} / {len(bvids_csv)}")

# Save the aligned image features to npy_data
np.save(os.path.join("npy_data", "picture_features_aligned.npy"), aligned_clip_feats)

