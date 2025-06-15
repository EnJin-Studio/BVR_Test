# 🎬 Bilibili Video View Count Prediction

This project aims to predict the view count of Bilibili homepage-recommended videos using multimodal input features, including thumbnail images, video titles, uploader information, and structured metadata. It integrates CLIP-based feature extraction, structured preprocessing, and regression model training for view prediction and video content scoring.

---

## 🚀 Project Objectives

- Crawl homepage video data from Bilibili (thumbnails, titles, uploader info, etc.)
- Extract semantic feature vectors using OpenAI’s CLIP model
- Integrate additional structured metadata (uploader followers, video duration, time since publication)
- Construct a unified 1039-dimensional input feature vector for each video
- Train a regression model to predict view counts
- Provide a scalable framework for content analysis, scoring, and recommendation

---

## 📁 Project Structure

```

bilibili-view-predictor/
├── data/                        # Feature storage
│   ├── clip\_image\_features.npy
│   ├── clip\_text\_features.npy
│   ├── uploader\_name\_features.npy
│   └── video\_meta.csv
├── images/                      # Downloaded thumbnails
├── scripts/                     # Core functionality
│   ├── bilibili_web_crawler.py  # Scrape homepage video & uploader info
│   ├── extract\_features.py     # Extract CLIP-based image & text features
│   ├── build\_dataset.py        # Combine all features into training data
│   └── train\_model.py          # Train MLP regression model
├── model/                       # Trained models
│   └── mlp\_regressor.pth
├── requirements.txt
└── README.md

````

---

## 🔢 Input Features (1039 Dimensions)

| Modality    | Feature                    | Dim | Description                                        |
|-------------|----------------------------|-----|----------------------------------------------------|
| Image       | Thumbnail embedding (CLIP) | 512 | Extracted using `CLIP.encode_image()`              |
| Text        | Title embedding (CLIP)     | 512 | Extracted using `CLIP.encode_text()`               |
| Text        | Uploader name embedding    | 12  | Encoded using CLIP (short-text semantic vector)    |
| Numeric     | Uploader follower count    | 1   | Log-transformed with `log1p(follower)`             |
| Numeric     | Video duration (in seconds)| 1   | Original video length                              |
| Numeric     | Time since published (days)| 1   | Current time minus upload timestamp                |

📐 **Total: 512 + 512 + 12 + 3 = 1039 features**

---

## 🎯 Prediction Target

- `view_count` — the number of video views  
- Transformed using `log1p(view_count)` to normalize extreme values

---

## 💻 Installation

```bash
pip install -r requirements.txt
````

Key dependencies include:

* `torch`, `clip-by-openai`, `pandas`, `numpy`, `requests`, `tqdm`

---

## 🛠️ Usage Instructions

### 1. Fetch homepage video data

```bash
python scripts/fetch_data.py
```

### 2. Extract CLIP-based features (image, title, uploader name)

```bash
python scripts/extract_features.py
```

### 3. Build the full dataset (1039D input, log view count as target)

```bash
python scripts/build_dataset.py
```

### 4. Train the regression model

```bash
python scripts/train_model.py
```

---

## 📊 Example Use Case

Once trained, the model can be used to:

* Estimate the expected view count of any new video
* Provide content creators with scoring feedback (image/text performance)
* Support a recommendation pipeline for homepage candidate ranking
* Identify high-potential videos before publishing

---

## 🧪 Planned Features (Backlog)

* Add more metadata (e.g., likes, favorites, video category tags)
* Visualize high-dimensional feature space via PCA/t-SNE
* Explore image-text attention fusion with cross-modal transformers
* Build a Streamlit-based demo for real-time content evaluation
* Extend model to multitask prediction (views + likes + comments)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Authors

**Ziyuan Chu (Eric)**  
Boston University · Computer Engineering  
Email: czyuan@bu.edu

**Feng Tai (Jimmy)**  
Boston University · Computer Engineering  
Email: jimmytai@bu.edu



