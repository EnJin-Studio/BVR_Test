import os
import csv
import requests
from PIL import Image
from io import BytesIO

# === 配置部分 ===
CSV_FILE = "bilibili_data_merged_deduped.csv"       # 输入CSV文件
IMAGE_DIR = "images"              # 图像保存目录
BVID_COL = "bvid"                 # 视频唯一ID列名
URL_COL = "pic"                   # 图片链接列名
TIMEOUT = 5                       # 请求超时秒数

def download_image(bvid, url, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{bvid}.jpg")

    if os.path.exists(save_path):
        print(f"[SKIP] {bvid} already downloaded.")
        return

    try:
        response = requests.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image.save(save_path)
        print(f"[OK] {bvid} downloaded.")
    except Exception as e:
        print(f"[ERROR] Failed to download {bvid}: {e}")

def main():
    with open(CSV_FILE, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            bvid = row.get(BVID_COL)
            url = row.get(URL_COL)
            if bvid and url:
                download_image(bvid, url, IMAGE_DIR)

if __name__ == "__main__":
    main()
