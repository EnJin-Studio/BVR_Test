from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import torch
import os

# Use GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device=device)

# Load video metadata CSV
df = pd.read_csv(os.path.join("csv_data", "video_meta.csv"))

# Encode titles
titles = df["title"].astype(str).tolist()
title_embeddings = model.encode(titles, show_progress_bar=True, batch_size=64, device=device)
title_array = np.array(title_embeddings)

# Encode uploader names
names = df["uploader_name"].astype(str).fillna("").tolist()
name_embeddings = model.encode(names, show_progress_bar=True, batch_size=64, device=device)
name_array = np.array(name_embeddings)

# Save to npy_data folder
np.save(os.path.join("npy_data", "title.npy"), title_array)
np.save(os.path.join("npy_data", "uploader.npy"), name_array)
