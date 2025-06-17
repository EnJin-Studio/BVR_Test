import re
import requests
import torch
import clip
import numpy as np
from PIL import Image
from io import BytesIO
from datetime import datetime
from transformers import BertTokenizer, BertModel
import torch.nn as nn

# === 配置 ===
CLIP_DEVICE = "cpu"
BERT_DEVICE = "cpu"
MODEL_PATH = "npy_data/mlp_fold4.pth"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# === MLP 模型定义 ===
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.model(x)

# === 提取 BVID ===
def extract_bvid(url):
    match = re.search(r"BV\w+", url)
    return match.group(0) if match else None

# === 获取视频元数据 ===
def fetch_video_info(bvid):
    api_url = "https://api.bilibili.com/x/web-interface/view"
    resp = requests.get(api_url, params={"bvid": bvid}, headers=HEADERS, timeout=5)
    resp.raise_for_status()
    return resp.json()["data"]

# === 获取粉丝数 ===
def fetch_follower_count(mid):
    url = "https://api.bilibili.com/x/relation/stat"
    resp = requests.get(url, params={"vmid": mid}, headers=HEADERS, timeout=5)
    resp.raise_for_status()
    return resp.json().get("data", {}).get("follower", 0)

# === 提取图像向量（512）===
def get_clip_image_vector(url):
    model, preprocess = clip.load("ViT-B/32", device=CLIP_DEVICE)
    image = Image.open(BytesIO(requests.get(url).content)).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(CLIP_DEVICE)
    with torch.no_grad():
        vec = model.encode_image(image_input).cpu().numpy()
    return vec.squeeze()

# === 提取文本向量（384）===
def get_bert_embedding(text):
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertModel.from_pretrained("bert-base-chinese").to(BERT_DEVICE)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(BERT_DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_embedding.squeeze()[:384]

# === 数值特征标准化（duration, follower, days_since, danmaku）===
def build_numeric_feature(duration, pubdate, followers, danmaku):
    seconds_ago = int(datetime.now().timestamp()) - pubdate
    raw = np.array([
        duration,
        followers,
        seconds_ago / 86400.0,
        danmaku
    ], dtype=np.float32)

    # 替换为你训练时的 scaler 均值/标准差
    mean = np.array([143.2, 1.7e6, 875.4, 24.6], dtype=np.float32)
    std  = np.array([82.1, 3.4e6, 594.3, 67.2], dtype=np.float32)
    z = (raw - mean) / std
    return z

# === 主预测函数 ===
def predict_from_url(url):
    bvid = extract_bvid(url)
    if not bvid:
        print("[ERROR] 无效的 Bilibili 链接。")
        return

    print(f"[INFO] 获取视频信息: {bvid}")
    data = fetch_video_info(bvid)

    image_vec = get_clip_image_vector(data["pic"])              # [512]
    title_vec = get_bert_embedding(data["title"])               # [384]
    uploader_vec = get_bert_embedding(data["owner"]["name"])    # [384]
    followers = fetch_follower_count(data["owner"]["mid"])      # int
    numeric_vec = build_numeric_feature(
        duration=data["duration"],
        pubdate=data["pubdate"],
        followers=followers,
        danmaku=data["stat"]["danmaku"]
    )  # [4]

    # 拼接特征
    x = np.hstack([image_vec, title_vec, uploader_vec, numeric_vec])
    assert x.shape[0] == 1284, f"[ERROR] 输入特征维度应为1284，实际为 {x.shape[0]}"

    x_tensor = torch.tensor(x[np.newaxis, :], dtype=torch.float32)

    # 加载模型
    model = MLPRegressor(input_dim=1284)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        y_log = model(x_tensor).item()
    y_pred = np.expm1(y_log)

    print(f"[PREDICTED] log(view_count) = {y_log:.4f}")
    print(f"[PREDICTED] estimated view_count ≈ {int(y_pred):,}")

# === 执行 ===
if __name__ == "__main__":
    test_url = input("请输入 B 站视频链接：\n> ").strip()
    predict_from_url(test_url)
