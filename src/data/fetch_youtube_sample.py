"""Download a small sample of YouTube comments to label by hand.

Usage:
    python src/data/fetch_youtube_sample.py VIDEO_ID [VIDEO_ID ...] [--per-video 35]

Writes data/external/youtube_sample.csv with the comment text only (no usernames).
Refuses to overwrite an existing file, so your labels are never lost.
"""
import argparse
import os
import random
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
KEY = os.getenv("YOUTUBE_API_KEY")
URL = "https://www.googleapis.com/youtube/v3/commentThreads"
OUT = ROOT / "data" / "external" / "youtube_sample.csv"


def fetch(video_id, pool=300):
    """Top-level comments, newest first, up to `pool` of them."""
    texts, token = [], None
    while len(texts) < pool:
        params = {"part": "snippet", "videoId": video_id, "maxResults": 100,
                  "order": "time", "textFormat": "plainText", "key": KEY}
        if token:
            params["pageToken"] = token
        r = requests.get(URL, params=params, timeout=30)
        if r.status_code != 200:
            msg = r.json().get("error", {}).get("message", r.text)
            print(f"  {video_id}: {msg}")
            break
        data = r.json()
        for item in data.get("items", []):
            texts.append(item["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
        token = data.get("nextPageToken")
        if not token:
            break
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_ids", nargs="+")
    parser.add_argument("--per-video", type=int, default=35)
    parser.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = parser.parse_args()

    if not KEY:
        raise SystemExit("YOUTUBE_API_KEY not found in .env")
    if OUT.exists() and not args.force:
        raise SystemExit(f"{OUT} already exists (it may contain your labels). Use --force to overwrite.")

    rng = random.Random(42)
    rows = []
    for vid in args.video_ids:
        print(f"fetching {vid}...")
        texts = fetch(vid)
        # tidy: collapse whitespace, drop empties and duplicates
        texts = list(dict.fromkeys(" ".join(t.split()) for t in texts if t.strip()))
        picked = rng.sample(texts, min(args.per_video, len(texts)))
        print(f"  {len(texts)} comments available, kept {len(picked)}")
        rows += [{"video_id": vid, "comment": t, "label": ""} for t in picked]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"wrote {len(rows)} comments to {OUT}")


if __name__ == "__main__":
    main()