import re
import requests
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime
from transformers import BertTokenizer, BertModel
import clip
from sentence_transformers import SentenceTransformer

torch.backends.cudnn.benchmark = True

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model/model_fold2_state_dict.pt"
HEADERS = {"User-Agent": "Mozilla/5.0"}

sbert_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device=device)
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# Get BVID
def extract_bvid(url):
    match = re.search(r"BV\w+", url)
    return match.group(0) if match else None

# Get video data
def fetch_video_info(bvid):
    api_url = "https://api.bilibili.com/x/web-interface/view"
    resp = requests.get(api_url, params={"bvid": bvid}, headers=HEADERS, timeout=5)
    resp.raise_for_status()
    return resp.json()["data"]

# Get follower count
def fetch_follower_count(mid):
    url = "https://api.bilibili.com/x/relation/stat"
    resp = requests.get(url, params={"vmid": mid}, headers=HEADERS, timeout=5)
    resp.raise_for_status()
    return resp.json().get("data", {}).get("follower", 0)

# Get embedding picture vector
def get_clip_image_vector(url):
    image = Image.open(BytesIO(requests.get(url).content)).convert("RGB")
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = clip_model.encode_image(image_input).cpu().numpy()
    return vec.squeeze()

# Get embedding sentence vector
def get_bert_embedding(text):
    vec = sbert_model.encode([text], convert_to_numpy=True)
    return vec.squeeze()

# Get duration, follower, days_since, danmaku
def build_numeric_feature(duration, pubdate, followers, danmaku):
    seconds_ago = int(datetime.now().timestamp()) - pubdate
    raw = np.array([
        duration,
        np.log1p(followers),
        np.log1p(seconds_ago),
    ], dtype=np.float32)

    scaler_mean = np.load("npy_data/scaler_mean.npy")
    scaler_std = np.load("npy_data/scaler_std.npy")
    z = (raw - scaler_mean) / scaler_std
    return z

class MultiModalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_branch = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3)
        )
        self.title_branch = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3)
        )
        self.uploader_branch = nn.Sequential(
            nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.3)
        )
        self.numeric_branch = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(), nn.Dropout(0.3)
        )
        self.fusion = nn.Sequential(
            nn.Linear(256 + 256 + 128 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, img, title, up, num):
        x1 = self.image_branch(img)
        x2 = self.title_branch(title)
        x3 = self.uploader_branch(up)
        x4 = self.numeric_branch(num)
        fused = torch.cat([x1, x2, x3, x4], dim=1)
        return self.fusion(fused)



# Predict main function
def predict_from_url(url):
    bvid = extract_bvid(url)
    if not bvid:
        print("[ERROR] 无效的 Bilibili 链接。")
        return

    print(f"[INFO] 获取视频信息: {bvid}")
    data = fetch_video_info(bvid)

    img_vec = get_clip_image_vector(data["pic"])                      # [512]
    title_vec = get_bert_embedding(data["title"])                     # [768]
    uploader_vec = get_bert_embedding(data["owner"]["name"])          # [768]
    followers = fetch_follower_count(data["owner"]["mid"])            # int
    numeric_vec = build_numeric_feature(
        duration=data["duration"],
        pubdate=data["pubdate"],
        followers=followers,
        danmaku=data["stat"]["danmaku"]
    )

    # Convert to tensor
    img_tensor = torch.tensor(img_vec[np.newaxis, :], dtype=torch.float32).to(device)
    title_tensor = torch.tensor(title_vec[np.newaxis, :], dtype=torch.float32).to(device)
    uploader_tensor = torch.tensor(uploader_vec[np.newaxis, :], dtype=torch.float32).to(device)
    numeric_tensor = torch.tensor(numeric_vec[np.newaxis, :], dtype=torch.float32).to(device)

    # Load model
    model = MultiModalMLP().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    with torch.no_grad():
        z_score_pred = model(img_tensor, title_tensor, uploader_tensor, numeric_tensor).item()

    log_mean = np.load("npy_data/view_log_mean.npy").item()
    log_std = np.load("npy_data/view_log_std.npy").item()
    log_view_count = z_score_pred * log_std + log_mean

    view_count = np.expm1(log_view_count)

    print(f"\n[PREDICTED] log(view_count) = {log_view_count:.4f}")
    print(f"[PREDICTED] estimated view_count ≈ {int(view_count):,}")


# Implement
if __name__ == "__main__":
    test_url = input("请输入 B 站视频链接：\n> ").strip()
    predict_from_url(test_url)
