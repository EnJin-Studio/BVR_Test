import os
import numpy as np

# Path configuration
NPY_DIR = "npy_data"
OUT_X = os.path.join(NPY_DIR, "X.npy")
OUT_Y = os.path.join(NPY_DIR, "y.npy")

# Load aligned input features
image_feat    = np.load(os.path.join(NPY_DIR, "clip_image_features_aligned.npy"))  # [N, 512]
title_feat    = np.load(os.path.join(NPY_DIR, "title_embeddings.npy"))             # [N, 768]
uploader_feat = np.load(os.path.join(NPY_DIR, "uploader_embeddings.npy"))          # [N, 12]
numeric_feat  = np.load(os.path.join(NPY_DIR, "numeric_features.npy"))             # [N, 3]

# Load target values
y = np.load(os.path.join(NPY_DIR, "view_count_log.npy"))                           # [N]

# Check sample consistency
N = image_feat.shape[0]
assert all(x.shape[0] == N for x in [title_feat, uploader_feat, numeric_feat, y]), \
    "[ERROR] Inconsistent number of samples across features or labels"

# Concatenate into final input feature X
X = np.hstack([image_feat, title_feat, uploader_feat, numeric_feat])  # [N, D]

# Save results
np.save(OUT_X, X)
np.save(OUT_Y, y)

print(f"[DONE] X saved to {OUT_X}, shape = {X.shape}")
print(f"[DONE] y saved to {OUT_Y}, shape = {y.shape}")
