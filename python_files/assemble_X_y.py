import os
import numpy as np

# === 配置路径 ===
NPY_DIR = "npy_data"
OUT_X = os.path.join(NPY_DIR, "X.npy")
OUT_Y = os.path.join(NPY_DIR, "y.npy")

# === 加载对齐后的输入特征 ===
image_feat    = np.load(os.path.join(NPY_DIR, "clip_image_features_aligned.npy"))  # [N, 512]
title_feat    = np.load(os.path.join(NPY_DIR, "title_embeddings.npy"))             # [N, 768]
uploader_feat = np.load(os.path.join(NPY_DIR, "uploader_embeddings.npy"))          # [N, 12]
numeric_feat  = np.load(os.path.join(NPY_DIR, "numeric_features.npy"))             # [N, 3]

# === 加载目标值 ===
y = np.load(os.path.join(NPY_DIR, "view_count_log.npy"))                           # [N]

# === 检查样本一致性 ===
N = image_feat.shape[0]
assert all(x.shape[0] == N for x in [title_feat, uploader_feat, numeric_feat, y]), \
    "[ERROR] Inconsistent number of samples across features or labels"

# === 拼接为最终输入特征 X ===
X = np.hstack([image_feat, title_feat, uploader_feat, numeric_feat])  # [N, D]

# === 保存结果 ===
np.save(OUT_X, X)
np.save(OUT_Y, y)

print(f"[DONE] X saved to {OUT_X}, shape = {X.shape}")
print(f"[DONE] y saved to {OUT_Y}, shape = {y.shape}")
