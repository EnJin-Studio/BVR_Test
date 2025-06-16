import numpy as np
import pandas as pd

# 1. Load the main CSV file used for model training
df = pd.read_csv("video_meta.csv")
bvids_csv = df["bvid"].astype(str).tolist()

# 2. Load original image features and their corresponding bvid order
clip_feats = np.load("clip_image_features.npy")   # shape: [M, 512]
clip_bvids = np.load("clip_image_bvids.npy")      # shape: [M]

# 3. Build a mapping from bvid to index
bvid_to_index = {bvid: idx for idx, bvid in enumerate(clip_bvids)}

# 4. Create aligned array, initialized with zero vectors
aligned_clip_feats = np.zeros((len(bvids_csv), clip_feats.shape[1]), dtype=np.float32)

missing = 0
for i, bvid in enumerate(bvids_csv):
    if bvid in bvid_to_index:
        aligned_clip_feats[i] = clip_feats[bvid_to_index[bvid]]
    else:
        aligned_clip_feats[i] = np.zeros(clip_feats.shape[1])
        missing += 1

print(f"[INFO] Alignment complete. Missing image features: {missing} / {len(bvids_csv)}")

# 5. Save the aligned image features
np.save("clip_image_features_aligned.npy", aligned_clip_feats)

