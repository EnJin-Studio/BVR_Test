import os
import csv
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# Configuration Section
CSV_FILE = os.path.join("csv_data", "video_meta.csv")   # Input CSV file
IMAGE_DIR = "images"                                    # Directory to save images
BVID_COL = "bvid"                  
URL_COL = "pic"                  
TIMEOUT = 5                                             # Request timeout
LOG_FILE = os.path.join("logs", "download_log.csv")     # Save download log

def download_image(bvid, url, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{bvid}.jpg")

    if os.path.exists(save_path):
        return "SKIP", f"{bvid},SKIP,already downloaded"

    try:
        if url.startswith("//"):
            url = "https:" + url
        response = requests.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image.save(save_path)
        return "OK", f"{bvid},OK,downloaded successfully"
    except Exception as e:
        return "ERROR", f"{bvid},ERROR,{e}"

def main():
    log_entries = []

    with open(CSV_FILE, newline='', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
        for row in tqdm(reader, desc="Downloading images"):
            bvid = row.get(BVID_COL)
            url = row.get(URL_COL)
            if bvid and url:
                status, log_line = download_image(bvid, url, IMAGE_DIR)
                tqdm.write(f"[{status}] {bvid}")
                log_entries.append(log_line)
    
    with open(LOG_FILE, "w", encoding="utf-8") as logf:
        logf.write("bvid,status,message\n")
        for line in log_entries:
            logf.write(line + "\n")
    print(f"\n[INFO] 日志已保存至 {LOG_FILE}")

if __name__ == "__main__":
    main()