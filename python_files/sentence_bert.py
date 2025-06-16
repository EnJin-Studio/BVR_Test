from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# Load the model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Read the CSV file
df = pd.read_csv("video_meta.csv")

# Encode titles
titles = df["title"].astype(str).tolist()
title_embeddings = model.encode(titles, show_progress_bar=True, batch_size=32)
title_array = np.array(title_embeddings)

# Encode uploader names
names = df["uploader_name"].astype(str).fillna("").tolist()
name_embeddings = model.encode(names, show_progress_bar=True, batch_size=32)
name_array = np.array(name_embeddings)

# Save as .npy files
np.save("title_embeddings.npy", title_array)
np.save("uploader_embeddings.npy", name_array)
