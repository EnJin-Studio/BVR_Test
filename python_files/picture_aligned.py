import numpy as np
import pandas as pd

# 1. 读取主 CSV 文件（你模型训练的主数据）
df = pd.read_csv("video_meta.csv")
bvids_csv = df["bvid"].astype(str).tolist()

# 2. 加载原始图像特征和其对应的 bvid 顺序
clip_feats = np.load("clip_image_features.npy")   # shape: [M, 512]
clip_bvids = np.load("clip_image_bvids.npy")      # shape: [M]

# 3. 建立 bvid → index 映射表
bvid_to_index = {bvid: idx for idx, bvid in enumerate(clip_bvids)}

# 4. 创建对齐后的数组，填零向量初始化
aligned_clip_feats = np.zeros((len(bvids_csv), clip_feats.shape[1]), dtype=np.float32)

missing = 0
for i, bvid in enumerate(bvids_csv):
    if bvid in bvid_to_index:
        aligned_clip_feats[i] = clip_feats[bvid_to_index[bvid]]
    else:
        aligned_clip_feats[i] = np.zeros(clip_feats.shape[1])
        missing += 1

print(f"[INFO] 对齐完成，共缺失图像特征的样本数：{missing} / {len(bvids_csv)}")

# 5. 保存对齐后的图像特征
np.save("clip_image_features_aligned.npy", aligned_clip_feats)
