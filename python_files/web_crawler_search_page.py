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

def parse_duration_to_seconds(duration_str):
    parts = duration_str.split(":")
    try:
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(int, parts)
            return m * 60 + s
        else:
            return 0
    except:
        return 0

# Simulate browser request to Bilibili homepage, fetch data from Bilibili API
# User-Agent represents the browser making the request
# Referer indicates that the request is from Bilibili
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
    "Referer": "https://search.bilibili.com",
    "Origin": "https://search.bilibili.com",
    "Accept": "application/json",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Cookie": "_uuid=810D1082B9-F1073-81106-59101-E759C98F3F6F91420infoc; buvid_fp=4efefeec27f4b6797c2d67e3f6431784; buvid3=5DCD4719-61DA-AB7F-DAB1-8E4264DDC26A05662infoc; b_nut=1749940793; buvid4=445080EE-6521-E55B-622B-BD87E123894305662-025061506-Csu38aAzj1bTMozpd574CA%3D%3D; header_theme_version=CLOSE; enable_web_push=DISABLE; enable_feed_channel=ENABLE; rpdid=|(ukR|ukYl))0J'u~RmmR)J|l; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NTAyNTkwMzMsImlhdCI6MTc0OTk5OTc3MywicGx0IjotMX0.M1su7Q6pBFjh5C2E3GJmWup2shxdFnr6nLfE8qs2lbA; bili_ticket_expires=1750258973; CURRENT_FNVAL=4048; b_lsid=558EB1068_1977A7FD606; home_feed_column=5; browser_resolution=1693-866"
}

# Fetch videos from Bilibili search
# max page is 50
def search_by_keyword(keyword, page):
    url = "https://api.bilibili.com/x/web-interface/search/type"
    params = {
        "search_type": "video",
        "keyword": keyword,
        "page": page
    }
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

# Extract required fields
def parse_search_items(data):
    items = []
    for v in data.get("data", {}).get("result", []):
        bvid = v.get("bvid")
        title = v.get("title")
        pic = v.get("pic")
        duration_str = v.get("duration", "0:00")
        duration = parse_duration_to_seconds(str(duration_str))
        pubdate_ts = v.get("pubdate")
        view_count = v.get("play")
        danmaku_count = int(v.get("video_review", "0"))
        uploader_name = v.get("author")
        mid = v.get("mid")

        items.append({
            "bvid": bvid,
            "title": clean_html(title),
            "pic": pic,
            "duration": duration,
            "pub_seconds_ago": int(time.time() - pubdate_ts) if pubdate_ts else None,
            "view_count": view_count,
            "danmaku_count": danmaku_count,
            "uploader_name": uploader_name,
            "uploader_mid": mid
        })
    return items

# Main logic
def main(keywords, pages_per_keyword=50, delay=1):
    history_file = os.path.join("csv_data", "video_meta.csv")
    seen = set()
    if os.path.exists(history_file):
        old_df = pd.read_csv(history_file)
        seen = set(old_df["bvid"].dropna().tolist())

    all_items = []

    for kw in keywords:
        print(f"\n[KEYWORD] {kw}")
        for page in range(1, pages_per_keyword + 1):
            print(f"[INFO] 抓取关键词第 {page} 页")
            try:
                resp = search_by_keyword(kw, page)
                new_items = parse_search_items(resp)
            except Exception as e:
                print(f"[ERROR] 抓取失败: {e}")
                continue

            count_before = len(all_items)
            for item in new_items:
                if item["bvid"] and item["bvid"] not in seen:
                    mid = item.get("uploader_mid")
                    item["uploader_follower"] = get_follower_count(mid) if mid else None
                    del item["uploader_mid"]
                    all_items.append(item)
                    seen.add(item["bvid"])
            print(f" → 新增 {len(all_items) - count_before} 条")
            time.sleep(random.uniform(delay, 2))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_filename = os.path.join("csv_data", f"data_batch_{timestamp}.csv")
    fieldnames = [
        "bvid", "title", "pic", "duration", "pub_seconds_ago",
        "view_count", "danmaku_count", "uploader_name", "uploader_follower"
    ]
    with open(batch_filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_items)
    print(f"[INFO] 批次保存为 {batch_filename}")

    # Combine with history
    csv_files = [batch_filename]
    if os.path.exists(history_file):
        csv_files.append(history_file)
    combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset="bvid", keep="first")
    combined_df.to_csv(history_file, index=False, encoding="utf-8")
    print(f"[DONE] 合并后共 {len(combined_df)} 条")

if __name__ == "__main__":
    all_keywords = [
        ["原神", "英雄联盟", "王者荣耀", "我的世界", "崩坏星穹铁道", "动画", "鬼畜", "进击的巨人", "舰队Collection", "cosplay"],
        ["电影解说", "剧情短片", "恐怖片", "日剧", "韩剧", "综艺", "配音", "明星", "采访", "悬疑片"],
        ["人工智能", "AI", "ChatGPT", "Python", "编程教学", "科技资讯", "自动驾驶", "电动车", "计算机视觉", "大模型"],
        ["高数", "英语学习", "考研", "数学", "历史", "法律", "医学知识", "科普", "论文写作", "公开课"],
        ["音乐", "舞蹈", "翻唱", "纯音乐", "电音", "乐器演奏", "ASMR", "声控", "合唱", "古风音乐"],
        ["vlog", "日常记录", "美食", "做饭", "搞笑", "情感分享", "宠物", "DIY", "手工", "生活技巧"]
    ]

    for group in all_keywords:
        main(group, pages_per_keyword=50, delay=1)
