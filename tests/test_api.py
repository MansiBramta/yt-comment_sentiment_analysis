import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # so "import backend.app" works

from fastapi.testclient import TestClient

import backend.app as api

client = TestClient(api.app)
VALID = {"negative", "neutral", "positive"}

FAKE_COMMENTS = [
    {"text": "great video, loved it", "author": "a", "likes": 5, "date": "2026-10-01"},
    {"text": "worst video ever", "author": "b", "likes": 2, "date": "2026-10-02"},
    {"text": "what is the topic of this", "author": "c", "likes": 9, "date": "2026-10-02"},
]


def test_health():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_valid_labels():
    comments = ["this video is amazing", "worst video ever", "what is the topic of this"]
    r = client.post("/predict", json={"comments": comments})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 3
    assert all(item["sentiment"] in VALID for item in results)


def test_predict_obvious_cases():
    r = client.post("/predict", json={"comments": ["this video is amazing", "worst video ever"]})
    labels = [item["sentiment"] for item in r.json()["results"]]
    assert labels == ["positive", "negative"]


def test_predict_rejects_bad_input():
    r = client.post("/predict", json={"wrong_field": []})
    assert r.status_code == 422          # FastAPI's "invalid request" code


def test_analyze_with_fake_youtube(monkeypatch):
    # replace the real YouTube call with our fake data
    monkeypatch.setattr(api, "fetch_comments", lambda video_id, max_comments=300: FAKE_COMMENTS)

    r = client.post("/analyze", json={"video_id": "anything"})
    assert r.status_code == 200
    data = r.json()

    assert data["total_comments"] == 3
    assert data["unique_commenters"] == 3
    assert set(data["sentiment_counts"]) == VALID
    assert sum(data["sentiment_counts"].values()) == 3
    assert 0 <= data["avg_sentiment_score"] <= 10
    assert len(data["top_comments"]) <= 10
    assert data["top_comments"][0]["likes"] == 9    # sorted by likes


def test_analyze_no_comments(monkeypatch):
    monkeypatch.setattr(api, "fetch_comments", lambda video_id, max_comments=300: [])
    r = client.post("/analyze", json={"video_id": "anything"})
    assert r.status_code == 404