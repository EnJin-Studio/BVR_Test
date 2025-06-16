import pandas as pd
import numpy as np

# 读取 CSV
df = pd.read_csv("video_meta.csv")

# 提取 view_count 并处理缺失值
df["view_count"] = df["view_count"].fillna(0)

# 做 log 变换（加1是为了避免 log(0) 的情况）
view_count_log = np.log1p(df["view_count"].astype(np.float32))  # log(1 + x)

# 保存为 .npy 文件
np.save("view_count_log.npy", view_count_log.values)
