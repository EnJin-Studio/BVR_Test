from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# 载入模型
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# 读取CSV文件
df = pd.read_csv("video_meta.csv")

# 对title编码
titles = df["title"].astype(str).tolist()
title_embeddings = model.encode(titles, show_progress_bar=True, batch_size=32)
title_array = np.array(title_embeddings)

# 对uploader_name编码
names = df["uploader_name"].astype(str).fillna("").tolist()
name_embeddings = model.encode(names, show_progress_bar=True, batch_size=32)
name_array = np.array(name_embeddings)

# 保存为npy格式
np.save("title_embeddings.npy", title_array)
np.save("uploader_embeddings.npy", name_array)
