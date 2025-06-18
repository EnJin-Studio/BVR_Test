# 🔍 Bilibili Video View Count Predictor

A multimodal machine learning pipeline to predict Bilibili homepage video view counts, integrating CLIP and Sentence-BERT embeddings with uploader and content metadata. Designed to support content ranking, scoring, and performance forecasting.

---

## 🎯 Project Goals

- 📥 Crawl Bilibili homepage video metadata and thumbnails  
- 🧠 Extract semantic features using OpenAI’s CLIP (image) & Sentence-BERT (title + uploader name)  
- 🧮 Combine numerical metadata (e.g., followers, duration, danmu count)  
- 🧷 Assemble unified 2052D feature vectors per video  
- 📈 Train and evaluate a regression model (MLP)  
- 📊 Provide predictions, content insights, and ranking capabilities  

---

## 📁 Project Structure

```
python_files/
├── csv_data/                       # CSV data files (e.g., video_meta.csv)
├── images/                         # Downloaded thumbnails
├── model/                          # Saved PyTorch models
├── npy_data/                       # Extracted features (CLIP, BERT, numeric)
│   └── ...
│
├── extract_numerical.py            # Extract numeric metadata
├── extract_view_count.py           # Extract and process view counts
├── picture_aligned.py              # Align images with metadata
├── picture_clip.py                 # Extract image features via CLIP
├── picture_download.py             # Download thumbnails
├── predict.py                      # Run trained model for inference
├── sentence_bert.py                # Encode uploader name and title via Sentence-BERT
├── train_model.py                  # Train MLP regression model
├── web_crawler_main_page.py        # crawler for bilibili homepage(video & uploader info)
├── web_crawler_search_page.py      # crawler for tag-based search(video & uploader info)
.gitignore
README.md
```

---

## 📌 Input Feature Vector (2052 Dimensions)

| Type     | Feature Name              | Dim | Description                                    |
|----------|---------------------------|-----|------------------------------------------------|
| Image    | Thumbnail (CLIP)          | 512 | `CLIP.encode_image()`                          |
| Text     | Title (CLIP)              | 768 | `SentenceTransformer.encode()`                 |
| Text     | Uploader Name (SBERT)     | 768 | `SentenceTransformer.encode()`                 |
| Numeric  | Followers (log1p scaled)  | 1   | `log1p(follower_count)`                        |
| Numeric  | Video Duration (sec)      | 1   | `Raw duration`                                 |
| Numeric  | Time Since Published (sec)| 1   | `log1p(seconds since upload)`                  |
| Numeric  | Danmu Count               | 1   | `log1p(current visible danmu bullet count)`    |

---

## 🎯 Prediction Target

- `view_count` (log1p-transformed to reduce skew)

---

## 🔧 Installation

Key dependencies:
- `torch`
- `openai-clip`
- `sentence-transformers`
- `numpy`, `pandas`
- `requests`, `tqdm`

---

## 🚀 Usage

### 1. Crawl Bilibili videos

```bash
python python_files/web_crawler_main_page.py
python python_files/web_crawler_search_page.py
```

### 2. Download thumbnails

```bash
python python_files/picture_download.py
```

### 3. Extract features

```bash
python python_files/picture_clip.py         # CLIP image
python python_files/picture_aligned.py      # aligned CLIP image with meta data
python python_files/sentence_bert.py        # title/uploader SBERT
python python_files/extract_numerical.py    # numeric metadata
```

### 4. Build output dataset

```bash
python python_files/extract_view_count.py
```

### 5. Train model

```bash
python python_files/train_model.py
```

### 6. Predict view count

```bash
python python_files/predict.py
```

---

## 🧠 Potential Applications

- Predict the popularity of a new video before upload  
- Assist Bilibili creators in thumbnail/title optimization  
- Rank candidate videos in a homepage recommender  
- Visualize latent features of popular content  

---

## 🧪 Future Work

- Incorporate tags, favorites, likes as auxiliary features  
- Use t-SNE or PCA to visualize embeddings  
- Explore multi-task learning (views + likes)  
- Build a Website for easier use  

---

## 👥 Authors

**Ziyuan Chu (Eric)**  
Boston University · Computer Engineering  
📧 [czyuan@bu.edu](mailto:czyuan@bu.edu)

**Feng Tai (Jimmy)**  
Boston University · Computer Engineering  
📧 [jimmytai@bu.edu](mailto:jimmytai@bu.edu)

---

## 📄 License

MIT License. Free to use with attribution.
