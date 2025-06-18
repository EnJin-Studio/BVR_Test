import time
import csv
import requests
import random
import os
import pandas as pd
from datetime import datetime
from html import unescape
import re

# Clean title
def clean_html(text):
    text = unescape(text)
    return re.sub(r'<[^>]+>', '', text)

# Simulate browser request to Bilibili homepage, fetch data from Bilibili API
# User-Agent represents the browser making the request
# Referer indicates that the request is from Bilibili
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
    "Referer": "https://www.bilibili.com"
}

# Fetch recommended videos from Bilibili homepage
# ps: number of items per request, suggest 30-50
# fresh_idx: index for pagination
# timeout waits for 5 seconds before throwing an exception
def fetch_recommend_page(ps=30, fresh_idx=1):
    url = "https://api.bilibili.com/x/web-interface/index/top/feed/rcmd"
    params = {"ps": ps, "fresh_idx": fresh_idx, "fresh_type": 4}
    resp = requests.get(url, params=params, headers=HEADERS, timeout=5)
    resp.raise_for_status()
    return resp.json()

# Get follower count for uploader (Bilibili separates video data from user data)
# mid is the uploader’s user ID; follower info is under data
def get_follower_count(mid):
    url = "https://api.bilibili.com/x/relation/stat"
    params = {"vmid": mid}
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", {}).get("follower", None)
    except Exception as e:
        print(f"[ERROR] 获取mid={mid}的粉丝数失败: {e}")
        return None

# Extract required fields from fetched recommend page
def parse_recommend_items(data):
    items = []
    for v in data.get("data", {}).get("item", []):
        bvid = v.get("bvid", None)
        title = v.get("title", None)
        pic = v.get("pic", None)
        duration = v.get("duration", None)
        pubdate_ts = v.get("pubdate", None)

        view_count = v.get("stat", {}).get("view")
        danmaku_count = v.get("stat", {}).get("danmaku")
        uploader_name = v.get("owner", {}).get("name")
        uploader_mid = v.get("owner", {}).get("mid")

        items.append({
            "bvid": bvid,
            "title": clean_html(title),
            "pic": pic,
            "duration": duration,
            "pub_seconds_ago": int(time.time() - pubdate_ts) if pubdate_ts else None,
            "view_count": view_count,
            "danmaku_count": danmaku_count,
            "uploader_name": uploader_name,
            "uploader_mid": uploader_mid
        })
    return items

# Main logic, pages: number of fetches; ps: items per fetch
def main(pages=10, ps=30, delay=1):
    
    # Try to load previously seen bvids
    history_file = os.path.join("python_files", "csv_data", "video_meta.csv")
    seen = set()
    if os.path.exists(history_file):
        old_df = pd.read_csv(history_file)
        seen = set(old_df["bvid"].dropna().tolist())


    all_items = []

    for idx in range(1, pages + 1):
        print(f"[INFO] 正在抓取第 {idx} 页")
        try:
            resp = fetch_recommend_page(ps=ps, fresh_idx=idx)
            new_items = parse_recommend_items(resp)
        except Exception as e:
            print(f"[ERROR] 抓取第 {idx} 页失败: {e}")
            continue

        count_before = len(all_items)

        # Append follower count and remove mid
        for item in new_items:
            if item["bvid"] and item["bvid"] not in seen:
                mid = item.get("uploader_mid")
                if mid:
                    item["uploader_follower"] = get_follower_count(mid)
                else:
                    item["uploader_follower"] = None
                del item["uploader_mid"]
                all_items.append(item)
                seen.add(item["bvid"])

        print(f" → 本页新增 {len(all_items) - count_before} 条（总计 {len(all_items)} 条）")
        time.sleep(random.uniform(delay, 2))

    # Save current batch to timestamped CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_filename = os.path.join("csv_data", f"data_batch_{timestamp}.csv")
    fieldnames = [
        "bvid", "title", "pic", "duration", "pubdate", "pub_seconds_ago",
        "view_count", "danmaku_count", "uploader_name", "uploader_follower"
    ]
    with open(batch_filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_items)

    print(f"[INFO] 本次批次保存为 {batch_filename}（{len(all_items)} 条）")

    # Auto-merge and deduplicate with history
    csv_files = [batch_filename]
    if os.path.exists(history_file):
        csv_files.append(history_file)

    combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset="bvid", keep="first")
    combined_df.to_csv(history_file, index=False, encoding="utf-8")

    print(f"[DONE] 已合并并保存为 {history_file}，总计 {len(combined_df)} 条唯一数据")

if __name__ == "__main__":
    main(pages=100, ps=30, delay=1)
