import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 读取 CSV
df = pd.read_csv("bilibili_data_merged_deduped.csv")

# 确保这些列都存在并处理缺失值
numeric_cols = ["duration", "pub_seconds_ago", "uploader_follower", "danmaku_count"]
df[numeric_cols] = df[numeric_cols].fillna(0)

# Z-score 标准化
scaler = StandardScaler()
numeric_features = scaler.fit_transform(df[numeric_cols].values.astype(np.float32))

# 保存为 .npy 文件
np.save("numeric_features.npy", numeric_features)
