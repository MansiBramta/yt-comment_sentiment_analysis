import os
import re
import joblib
import nltk
import requests
from collections import Counter, defaultdict
from pathlib import Path
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# ---- paths, keys, model ----
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
YT_KEY = os.getenv("YOUTUBE_API_KEY")

model = joblib.load(ROOT / "models" / "stacking_model.pkl")
vectorizer = joblib.load(ROOT / "models" / "tfidf_vectorizer.pkl")

LABELS = {-1: "negative", 0: "neutral", 1: "positive"}

# ---- same cleaning as training ----
KEEP_WORDS = {"not", "but", "however", "no", "yet"}
STOP_WORDS = set(stopwords.words("english")) - KEEP_WORDS
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("\n", " ")
    text = re.sub(r"[^A-Za-z0-9\s!?.,]", "", text)
    words = [w for w in text.split() if w not in STOP_WORDS]
    words = [LEMMATIZER.lemmatize(w) for w in words]
    return " ".join(words)


# ---- API setup ----
app = FastAPI(title="YT Sentiment API")

# lets the Chrome extension call this API
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


class Comments(BaseModel):
    comments: list[str]


class Video(BaseModel):
    video_id: str


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(body: Comments):
    cleaned = [clean_text(c) for c in body.comments]
    X = vectorizer.transform(cleaned)
    preds = model.predict(X)
    return {"results": [
        {"comment": c, "sentiment": LABELS[int(p)]}
        for c, p in zip(body.comments, preds)
    ]}


# ---- YouTube comments ----
def fetch_comments(video_id: str, max_comments: int = 300):
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    items, token = [], None
    while len(items) < max_comments:
        params = {"part": "snippet", "videoId": video_id, "maxResults": 100,
                  "textFormat": "plainText", "key": YT_KEY}
        if token:
            params["pageToken"] = token
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            msg = r.json().get("error", {}).get("message", "YouTube API error")
            raise HTTPException(status_code=r.status_code, detail=msg)
        data = r.json()
        for it in data.get("items", []):
            s = it["snippet"]["topLevelComment"]["snippet"]
            items.append({"text": s["textDisplay"], "author": s["authorDisplayName"],
                          "likes": s.get("likeCount", 0), "date": s["publishedAt"][:10]})
        token = data.get("nextPageToken")
        if not token:
            break
    return items[:max_comments]


@app.post("/analyze")
def analyze(body: Video):
    items = fetch_comments(body.video_id)
    if not items:
        raise HTTPException(status_code=404, detail="No comments found")

    texts = [i["text"] for i in items]
    cleaned = [clean_text(t) for t in texts]
    preds = model.predict(vectorizer.transform(cleaned))
    for i, p in zip(items, preds):
        i["sentiment"] = LABELS[int(p)]

    counts = Counter(i["sentiment"] for i in items)

    by_date = defaultdict(Counter)
    for i in items:
        by_date[i["date"]][i["sentiment"]] += 1
    trend = [{"date": d, **{k: by_date[d].get(k, 0) for k in LABELS.values()}}
             for d in sorted(by_date)]

    top = sorted(items, key=lambda i: i["likes"], reverse=True)[:10]
    words = Counter(w for c in cleaned for w in c.split() if len(w) > 2)
    mean = sum(int(p) for p in preds) / len(preds)

    return {
        "total_comments": len(items),
        "unique_commenters": len({i["author"] for i in items}),
        "avg_comment_length": round(sum(len(t.split()) for t in texts) / len(texts), 2),
        "avg_sentiment_score": round((mean + 1) * 5, 2),   # 0 to 10
        "sentiment_counts": {k: counts.get(k, 0) for k in LABELS.values()},
        "trend": trend,
        "top_comments": [{"text": i["text"], "likes": i["likes"],
                          "sentiment": i["sentiment"]} for i in top],
        "word_counts": [{"word": w, "count": c} for w, c in words.most_common(50)],
    }