const API = "http://127.0.0.1:8000";
const $ = (id) => document.getElementById(id);

function getVideoId(url) {
  try {
    const u = new URL(url);
    if (u.hostname.includes("youtube.com") && u.searchParams.get("v")) {
      return u.searchParams.get("v");
    }
    if (u.hostname === "youtu.be") return u.pathname.slice(1);
  } catch (e) {}
  return null;
}

function drawPie(c) {
  new Chart($("pie"), {
    type: "pie",
    data: {
      labels: ["Positive", "Neutral", "Negative"],
      datasets: [{
        data: [c.positive, c.neutral, c.negative],
        backgroundColor: ["#2ecc71", "#9aa0a6", "#e74c3c"],
        borderWidth: 0,
      }],
    },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#eee" } } },
    },
  });
}

function weekStart(dateStr) {
  const d = new Date(dateStr + "T00:00:00Z");
  const day = (d.getUTCDay() + 6) % 7;   // Monday = 0
  d.setUTCDate(d.getUTCDate() - day);
  return d.toISOString().slice(0, 10);
}

function groupByWeek(trend) {
  const weeks = {};
  trend.forEach((t) => {
    const w = weekStart(t.date);
    if (!weeks[w]) weeks[w] = { date: w, positive: 0, neutral: 0, negative: 0 };
    weeks[w].positive += t.positive;
    weeks[w].neutral += t.neutral;
    weeks[w].negative += t.negative;
  });
  return Object.values(weeks).sort((a, b) => a.date.localeCompare(b.date));
}

function drawTrend(trend) {
  new Chart($("trend"), {
    type: "line",
    data: {
      labels: trend.map((t) => t.date),
      datasets: [
        { label: "Positive", data: trend.map((t) => t.positive),
          borderColor: "#2ecc71", backgroundColor: "#2ecc71", tension: 0.3 },
        { label: "Neutral", data: trend.map((t) => t.neutral),
          borderColor: "#9aa0a6", backgroundColor: "#9aa0a6", tension: 0.3 },
        { label: "Negative", data: trend.map((t) => t.negative),
          borderColor: "#e74c3c", backgroundColor: "#e74c3c", tension: 0.3 },
      ],
    },
    options: {
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: "#ccc", maxRotation: 60 }, grid: { color: "#333" } },
        y: { beginAtZero: true, ticks: { color: "#ccc", precision: 0 }, grid: { color: "#333" } },
      },
      plugins: { legend: { labels: { color: "#eee" } } },
    },
  });
}

function renderTopComments(list) {
  const box = $("topComments");
  box.innerHTML = "";
  list.forEach((c) => {
    const card = document.createElement("div");
    card.className = "comment";

    const text = document.createElement("p");
    text.textContent = c.text;              // textContent keeps comments safe

    const meta = document.createElement("div");
    meta.className = "meta";

    const badge = document.createElement("span");
    badge.className = "badge " + c.sentiment;
    badge.textContent = c.sentiment;

    const likes = document.createElement("span");
    likes.textContent = "👍 " + c.likes;

    meta.append(badge, likes);
    card.append(text, meta);
    box.append(card);
  });
}

function renderCloud(words) {
  const box = $("cloud");
  box.innerHTML = "";
  const top = words.slice(0, 40);
  if (!top.length) return;

  const counts = top.map((w) => w.count);
  const min = Math.min(...counts);
  const max = Math.max(...counts);
  const colors = ["#3fa7ff", "#2ecc71", "#f1c40f", "#e67e22", "#9b59b6", "#1abc9c"];

  // shuffle so big words aren't all at the start
  const shuffled = [...top].sort(() => Math.random() - 0.5);
  shuffled.forEach((w, i) => {
    const span = document.createElement("span");
    span.textContent = w.word;
    const scale = max === min ? 0.5 : (w.count - min) / (max - min);
    span.style.fontSize = 12 + scale * 30 + "px";
    span.style.color = colors[i % colors.length];
    span.title = w.count + " times";
    box.append(span);
  });
}

async function main() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const id = getVideoId(tab.url);
  if (!id) {
    $("status").textContent = "Open a YouTube video first.";
    return;
  }
  $("videoId").textContent = id;
  $("status").textContent = "Fetching comments and analyzing...";

  try {
    const res = await fetch(`${API}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: id }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");

    $("status").textContent = "";
    $("total").textContent = data.total_comments;
    $("unique").textContent = data.unique_commenters;
    $("length").textContent = data.avg_comment_length + " words";
    $("score").textContent = data.avg_sentiment_score + "/10";
    drawPie(data.sentiment_counts);
    drawTrend(groupByWeek(data.trend));
    renderCloud(data.word_counts);
    renderTopComments(data.top_comments);
  } catch (e) {
    $("status").textContent = "Error: " + e.message;
  }
}

main();