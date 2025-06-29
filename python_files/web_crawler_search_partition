import time
import csv
import requests
import random
import os
import pandas as pd
from datetime import datetime
from html import unescape
import re
import math

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

# Partition List
PARTITIONS = {
    1: "动画",
    3: "音乐",
    4: "游戏",
    5: "娱乐",
    11: "番剧",
    13: "纪录片",
    23: "电视剧",
    36: "知识",
    83: "电影",
    119: "国创",
    129: "动物圈",
    155: "动画综合",
    160: "生活",
    167: "舞蹈",
    177: "鬼畜",
    181: "知识综合",
    188: "运动",
    211: "时尚",
    217: "科技",
    223: "美食",
    234: "汽车"
}


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
def parse_partition_items(vlist):
    items = []
    for v in vlist:
        aid = v.get("aid")
        title = v.get("title")
        pic = v.get("pic")
        duration = v.get("duration", 0)
        pubdate_ts = v.get("pubdate")
        view_count = v.get("stat", {}).get("view")
        danmaku_count = v.get("stat", {}).get("danmaku")
        uploader_name = v.get("owner", {}).get("name")
        mid = v.get("owner", {}).get("mid")

        items.append({
            "aid": aid,
            "title": clean_html(title),
            "pic": pic,
            "duration": duration,
            "pub_seconds_ago": int(time.time() - pubdate_ts) if pubdate_ts else None,
            "view_count": view_count,
            "danmaku_count": danmaku_count,
            "uploader_name": uploader_name,
            "uploader_mid": mid
        })
    print(f"[DEBUG] parse_partition_items 提取 {len(items)} 条视频")
    return items

def crawl_partition(rid, name, delay=1):
    print(f"\n📥 正在爬取分区 [{name}]")
    seen = set()
    all_items = []
    max_page = 100  # 默认最多100页

    for pn in range(1, max_page + 1):
        print(f" → 第 {pn} 页")
        url = f"https://api.bilibili.com/x/web-interface/dynamic/region?rid={rid}&pn={pn}&ps=50"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            print(f"[DEBUG] 请求 URL: {url}")
            print(f"[DEBUG] 响应状态码: {resp.status_code}")
            data = resp.json()
        except Exception as e:
            print(f"[ERROR] 请求或解析 JSON 失败: {e}")
            break

        data_obj = data.get("data", None)
        if not data_obj or "archives" not in data_obj:
            print("[DEBUG] 本页无数据或格式异常，跳出循环")
            break

        if pn == 1:
            count = data_obj.get("page", {}).get("count", 0)
            size = data_obj.get("page", {}).get("size", 50)
            if count and size:
                max_page = math.ceil(count / size)
                print(f"[DEBUG] 总视频数: {count}，每页: {size}，总页数: {max_page}")

        vlist = data_obj.get("archives", [])
        print(f"[DEBUG] 本页视频数量: {len(vlist)}")
        if vlist:
            print(f"[DEBUG] 示例视频标题: {vlist[0].get('title')}")

        new_items = parse_partition_items(vlist)
        print(f"[DEBUG] parse_partition_items 提取 {len(new_items)} 条视频")

        for item in new_items:
            aid = item["aid"]
            if aid and aid not in seen:
                item["uploader_follower"] = None
                del item["uploader_mid"]
                all_items.append(item)
                seen.add(aid)

        print(f" → 当前总数: {len(all_items)}")
        time.sleep(random.uniform(delay, 2))

    return all_items


def main():
    os.makedirs("csv_data", exist_ok=True)
    all_data = []

    for rid, name in PARTITIONS.items():
        part_items = crawl_partition(rid, name)
        if part_items:
            df = pd.DataFrame(part_items)
            df.to_csv(f"csv_data/{name}_{rid}.csv", index=False, encoding="utf-8-sig")
            all_data.extend(part_items)
            print(f"✅ 分区 [{name}] 完成，已保存 {len(part_items)} 条")

    df_all = pd.DataFrame(all_data)
    df_all.drop_duplicates(subset="aid", inplace=True)
    df_all.to_csv("csv_data/all_partitions_combined.csv", index=False, encoding="utf-8-sig")
    print(f"\n🎉 所有分区合并完成，共 {len(df_all)} 条")

if __name__ == "__main__":
    main()
