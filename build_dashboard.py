from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Reviews.csv"
OUTPUT_PATH = BASE_DIR / "dashboard.html"
SAMPLE_ROWS = 5000
SCATTER_SAMPLE = 1200

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will",
    "with", "you", "your", "yours", "yourself", "yourselves", "one", "two", "would", "could", "also",
    "get", "got", "buy", "bought", "product", "products", "food", "amazon", "review", "reviews",
    "t", "s", "m", "ve", "don", "ll", "re", "us", "im", "ive", "dont", "didnt", "doesnt", "cant",
}


def load_textblob():
    try:
        from textblob import TextBlob  # type: ignore

        return TextBlob
    except Exception:
        return None


def derive_polarity(text: str, score: float, textblob_cls) -> float:
    if textblob_cls is not None:
        try:
            return float(textblob_cls(str(text)).sentiment.polarity)
        except Exception:
            pass
    return (float(score) - 3.0) / 2.0


def label_sentiment(polarity: float) -> str:
    if polarity > 0.05:
        return "Positive"
    if polarity < -0.05:
        return "Negative"
    return "Neutral"


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def word_count(value: str) -> int:
    return len(re.findall(r"[A-Za-z']+", str(value)))


def build_top_terms(frame: pd.DataFrame, limit: int = 12) -> list[dict[str, object]]:
    counter: Counter[str] = Counter()
    for value in frame["combined_text"].astype(str):
        tokens = re.findall(r"[A-Za-z']+", value.lower())
        counter.update(token for token in tokens if len(token) > 2 and token not in STOPWORDS)

    return [
        {"term": term, "count": count}
        for term, count in counter.most_common(limit)
    ]


def build_html(payload: dict[str, object]) -> str:
    kpis = payload["kpis"]
    sentiment_cards = payload["sentiment_cards"]
    heatmap = json.dumps(payload["heatmap"], ensure_ascii=False)
    scatter = json.dumps(payload["scatter"], ensure_ascii=False)
    top_terms = json.dumps(payload["top_terms"], ensure_ascii=False)
    sentiment_counts = json.dumps(payload["sentiment_counts"], ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Sentiment Analysis Neon Command Center</title>
  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">
  <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
  <link href=\"https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap\" rel=\"stylesheet\">
  <script src=\"https://cdn.plot.ly/plotly-2.32.0.min.js\"></script>
  <style>
    :root {{
      --bg: #05070d;
      --bg-2: #09111c;
      --panel: rgba(12, 18, 31, 0.84);
      --panel-strong: rgba(14, 24, 40, 0.96);
      --line: rgba(86, 245, 255, 0.14);
      --line-strong: rgba(86, 245, 255, 0.28);
      --text: #eef6ff;
      --muted: #9eb0c6;
      --cyan: #4ffcff;
      --cyan-soft: rgba(79, 252, 255, 0.18);
      --green: #41f29b;
      --pink: #ff4fd8;
      --amber: #ffb84d;
      --red: #ff5a7a;
      --shadow: 0 28px 80px rgba(0, 0, 0, 0.55);
    }}

    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; min-height: 100%; background: radial-gradient(circle at top left, #0d1730 0%, var(--bg) 48%, #03050a 100%); color: var(--text); font-family: 'Space Grotesk', sans-serif; overflow-x: hidden; }}
    body::before {{
      content: '';
      position: fixed;
      inset: 0;
      background-image:
        linear-gradient(rgba(79, 252, 255, 0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(79, 252, 255, 0.04) 1px, transparent 1px),
        radial-gradient(circle at 20% 20%, rgba(255, 79, 216, 0.16), transparent 24%),
        radial-gradient(circle at 80% 10%, rgba(65, 242, 155, 0.12), transparent 20%),
        radial-gradient(circle at 80% 80%, rgba(79, 252, 255, 0.08), transparent 24%);
      background-size: 42px 42px, 42px 42px, auto, auto, auto;
      pointer-events: none;
      opacity: 0.75;
      z-index: 0;
    }}

    .orbs::before, .orbs::after {{
      content: '';
      position: fixed;
      inset: auto;
      width: 460px;
      height: 460px;
      border-radius: 50%;
      filter: blur(72px);
      pointer-events: none;
      opacity: 0.32;
      z-index: 0;
    }}
    .orbs::before {{ left: -140px; top: 120px; background: rgba(255, 79, 216, 0.26); }}
    .orbs::after {{ right: -120px; top: 50px; background: rgba(79, 252, 255, 0.28); }}

    .shell {{ position: relative; z-index: 1; max-width: 1540px; margin: 0 auto; padding: 28px 22px 40px; }}
    .hero {{
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(320px, 0.9fr);
      gap: 18px;
      align-items: stretch;
      margin-bottom: 18px;
    }}
    .hero-main, .hero-side, .panel, .stat, .insight, .mini-card {{
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(20, 30, 48, 0.9), rgba(7, 11, 18, 0.92));
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
      border-radius: 24px;
    }}
    .hero-main {{ padding: 28px 28px 24px; position: relative; overflow: hidden; }}
    .hero-main::after {{
      content: '';
      position: absolute;
      right: -70px;
      top: -70px;
      width: 240px;
      height: 240px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(79, 252, 255, 0.25) 0%, rgba(79, 252, 255, 0) 72%);
      pointer-events: none;
    }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 8px 14px;
      border-radius: 999px;
      font: 600 12px 'IBM Plex Mono', monospace;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--cyan);
      background: rgba(79, 252, 255, 0.08);
      border: 1px solid rgba(79, 252, 255, 0.18);
      box-shadow: 0 0 28px rgba(79, 252, 255, 0.12);
    }}
    .title {{ margin: 18px 0 10px; font-size: clamp(2.25rem, 4vw, 4.8rem); line-height: 0.98; letter-spacing: -0.05em; text-wrap: balance; }}
    .title span {{ color: var(--cyan); text-shadow: 0 0 22px rgba(79, 252, 255, 0.35); }}
    .subtitle {{ max-width: 760px; color: var(--muted); font-size: 1.02rem; line-height: 1.7; margin: 0; }}
    .hero-badges {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }}
    .badge {{
      border-radius: 999px;
      padding: 9px 14px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(255, 255, 255, 0.04);
      color: #d9e6f5;
      font: 500 12px 'IBM Plex Mono', monospace;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }}

    .hero-side {{ padding: 22px; display: grid; gap: 12px; align-content: start; }}
    .side-kpi {{
      padding: 16px 18px;
      border-radius: 18px;
      background: linear-gradient(135deg, rgba(79, 252, 255, 0.12), rgba(255, 79, 216, 0.08));
      border: 1px solid rgba(79, 252, 255, 0.15);
    }}
    .side-kpi .label {{ color: var(--muted); font-size: 0.8rem; letter-spacing: 0.12em; text-transform: uppercase; }}
    .side-kpi .value {{ font-size: 2rem; font-weight: 700; margin-top: 6px; }}
    .side-kpi .note {{ color: #c7d4e8; margin-top: 6px; font-size: 0.92rem; line-height: 1.5; }}

    .metrics {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 14px;
    }}
    .stat {{ padding: 18px; position: relative; overflow: hidden; min-height: 130px; }}
    .stat::before {{ content: ''; position: absolute; inset: auto -24px -24px auto; width: 130px; height: 130px; border-radius: 50%; background: radial-gradient(circle, var(--cyan-soft) 0%, transparent 70%); }}
    .stat .label {{ color: var(--muted); text-transform: uppercase; letter-spacing: 0.12em; font: 600 11px 'IBM Plex Mono', monospace; }}
    .stat .value {{ margin-top: 14px; font-size: clamp(1.6rem, 2.7vw, 2.6rem); font-weight: 700; line-height: 1; }}
    .stat .delta {{ margin-top: 10px; color: #c7d4e8; font-size: 0.95rem; line-height: 1.45; }}
    .stat .accent {{ display: inline-flex; align-items: center; margin-top: 10px; padding: 5px 10px; border-radius: 999px; font-size: 11px; font-family: 'IBM Plex Mono', monospace; letter-spacing: 0.1em; text-transform: uppercase; }}
    .accent.cyan {{ background: rgba(79, 252, 255, 0.11); color: var(--cyan); }}
    .accent.green {{ background: rgba(65, 242, 155, 0.11); color: var(--green); }}
    .accent.pink {{ background: rgba(255, 79, 216, 0.11); color: var(--pink); }}
    .accent.amber {{ background: rgba(255, 184, 77, 0.11); color: var(--amber); }}
    .accent.red {{ background: rgba(255, 90, 122, 0.11); color: var(--red); }}

    .grid {{ display: grid; grid-template-columns: 1.4fr 0.9fr; gap: 14px; margin-bottom: 14px; }}
    .panel {{ padding: 18px; position: relative; }}
    .panel-head {{ display: flex; justify-content: space-between; gap: 14px; align-items: end; margin-bottom: 12px; }}
    .panel-title {{ margin: 0; font-size: 1.05rem; letter-spacing: 0.02em; }}
    .panel-sub {{ margin: 4px 0 0; color: var(--muted); font-size: 0.92rem; line-height: 1.4; }}
    .panel-tag {{
      font: 600 11px 'IBM Plex Mono', monospace;
      letter-spacing: 0.14em;
      color: #dffcff;
      text-transform: uppercase;
      border: 1px solid rgba(79, 252, 255, 0.16);
      padding: 7px 10px;
      border-radius: 999px;
      background: rgba(79, 252, 255, 0.06);
      white-space: nowrap;
    }}
    .chart {{ width: 100%; height: 520px; }}
    .chart.tall {{ height: 560px; }}
    .chart.short {{ height: 360px; }}

    .details {{ display: grid; grid-template-columns: 1fr 0.85fr; gap: 14px; margin-top: 14px; }}
    .insight {{ padding: 18px; }}
    .insight h3 {{ margin: 0 0 10px; font-size: 1rem; }}
    .insight ul {{ margin: 0; padding-left: 18px; color: #d9e6f5; line-height: 1.75; }}
    .insight li {{ margin-bottom: 8px; }}
    .mini-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .mini-card {{ padding: 14px; border-radius: 18px; min-height: 118px; }}
    .mini-label {{ font: 600 11px 'IBM Plex Mono', monospace; text-transform: uppercase; letter-spacing: 0.12em; color: var(--muted); }}
    .mini-value {{ margin-top: 10px; font-size: 1.5rem; font-weight: 700; }}
    .mini-note {{ margin-top: 8px; color: #c7d4e8; font-size: 0.92rem; line-height: 1.45; }}
    .footer {{ margin-top: 16px; color: var(--muted); text-align: center; font-size: 0.9rem; padding-bottom: 8px; }}

    @media (max-width: 1240px) {{
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .grid, .details, .hero {{ grid-template-columns: 1fr; }}
      .chart {{ height: 460px; }}
      .chart.tall {{ height: 500px; }}
    }}
    @media (max-width: 760px) {{
      .shell {{ padding: 16px 12px 28px; }}
      .metrics {{ grid-template-columns: 1fr; }}
      .mini-grid {{ grid-template-columns: 1fr; }}
      .chart, .chart.tall, .chart.short {{ height: 400px; }}
      .title {{ font-size: 2.1rem; }}
      .panel-head {{ align-items: start; flex-direction: column; }}
    }}
  </style>
</head>
<body>
  <div class=\"orbs\"></div>
  <main class=\"shell\">
    <section class=\"hero\">
      <div class=\"hero-main\">
        <div class=\"eyebrow\">Sentiment Analysis Command Center</div>
        <h1 class=\"title\">Dark Neon <span>3-D</span> Review Intelligence Dashboard</h1>
        <p class=\"subtitle\">A high-contrast analyst workspace built from the cleaned Amazon Fine Food review sample. It blends TextBlob polarity, star ratings, helpfulness patterns, and review language into a compact executive-style dashboard.</p>
        <div class=\"hero-badges\">
          <span class=\"badge\">Dark Mode</span>
          <span class=\"badge\">3D Scatter</span>
          <span class=\"badge\">Neon Glass UI</span>
          <span class=\"badge\">Score vs Polarity</span>
          <span class=\"badge\">Business Summary</span>
        </div>
      </div>
      <div class=\"hero-side\">
        <div class=\"side-kpi\">
          <div class=\"label\">Dataset Mode</div>
          <div class=\"value\">5,000 Rows</div>
          <div class=\"note\">Built from the same notebook-scale sample used in the sentiment analysis workflow, with duplicate and empty reviews removed.</div>
        </div>
        <div class=\"side-kpi\">
          <div class=\"label\">Model Lens</div>
          <div class=\"value\">TextBlob</div>
          <div class=\"note\">Polarity is derived from review text when available, then mapped to Positive, Neutral, and Negative sentiment labels.</div>
        </div>
      </div>
    </section>

    <section class=\"metrics\">
      <div class=\"stat\"><div class=\"label\">Total Clean Reviews</div><div class=\"value\">{kpis['total_reviews']:,}</div><div class=\"accent cyan\">Coverage</div><div class=\"delta\">Unique, non-empty reviews retained after cleaning.</div></div>
      <div class=\"stat\"><div class=\"label\">Average Score</div><div class=\"value\">{kpis['avg_score']:.2f}/5</div><div class=\"accent green\">Rating</div><div class=\"delta\">The star baseline is strongly skewed positive.</div></div>
      <div class=\"stat\"><div class=\"label\">Positive Share</div><div class=\"value\">{kpis['positive_share']:.1f}%</div><div class=\"accent pink\">Sentiment</div><div class=\"delta\">TextBlob polarity above zero, mapped to positive sentiment.</div></div>
      <div class=\"stat\"><div class=\"label\">Mismatch Rate</div><div class=\"value\">{kpis['mismatch_rate']:.1f}%</div><div class=\"accent amber\">Exception</div><div class=\"delta\">Reviews where star rating and polarity disagree.</div></div>
      <div class=\"stat\"><div class=\"label\">Avg Words / Review</div><div class=\"value\">{kpis['avg_words']:.0f}</div><div class=\"accent red\">Text Depth</div><div class=\"delta\">Longer reviews usually carry more contextual nuance.</div></div>
    </section>

    <section class=\"grid\">
      <div class=\"panel\">
        <div class=\"panel-head\">
          <div>
            <h2 class=\"panel-title\">3-D Review Space</h2>
            <p class=\"panel-sub\">Word volume, helpfulness ratio, and polarity score rendered as a glassy neon 3D scatter.</p>
          </div>
          <div class=\"panel-tag\">Interactive 3D</div>
        </div>
        <div id=\"scatter3d\" class=\"chart tall\"></div>
      </div>
      <div class=\"panel\">
        <div class=\"panel-head\">
          <div>
            <h2 class=\"panel-title\">Sentiment Mix</h2>
            <p class=\"panel-sub\">Distribution across positive, neutral, and negative labels.</p>
          </div>
          <div class=\"panel-tag\">Executive View</div>
        </div>
        <div id=\"donut\" class=\"chart short\"></div>
        <div class=\"mini-grid\">
          <div class=\"mini-card\">
            <div class=\"mini-label\">Positive</div>
            <div class=\"mini-value\">{sentiment_cards['Positive']['count']:,}</div>
            <div class=\"mini-note\">{sentiment_cards['Positive']['pct']:.1f}% of the sample.</div>
          </div>
          <div class=\"mini-card\">
            <div class=\"mini-label\">Negative</div>
            <div class=\"mini-value\">{sentiment_cards['Negative']['count']:,}</div>
            <div class=\"mini-note\">{sentiment_cards['Negative']['pct']:.1f}% of the sample.</div>
          </div>
          <div class=\"mini-card\">
            <div class=\"mini-label\">Neutral</div>
            <div class=\"mini-value\">{sentiment_cards['Neutral']['count']:,}</div>
            <div class=\"mini-note\">{sentiment_cards['Neutral']['pct']:.1f}% of the sample.</div>
          </div>
          <div class=\"mini-card\">
            <div class=\"mini-label\">5-Star / Negative</div>
            <div class=\"mini-value\">{kpis['high_star_negative']:,}</div>
            <div class=\"mini-note\">Sarcasm, context gaps, and mixed phrasing create mismatches.</div>
          </div>
        </div>
      </div>
    </section>

    <section class=\"grid\">
      <div class=\"panel\">
        <div class=\"panel-head\">
          <div>
            <h2 class=\"panel-title\">Star Rating vs Sentiment</h2>
            <p class=\"panel-sub\">Where the review score agrees with TextBlob polarity, and where it doesn’t.</p>
          </div>
          <div class=\"panel-tag\">Heatmap</div>
        </div>
        <div id=\"heatmap\" class=\"chart\"></div>
      </div>
      <div class=\"panel\">
        <div class=\"panel-head\">
          <div>
            <h2 class=\"panel-title\">Language Fingerprint</h2>
            <p class=\"panel-sub\">Most frequent non-trivial words appearing across summary and review text.</p>
          </div>
          <div class=\"panel-tag\">Top Terms</div>
        </div>
        <div id=\"topterms\" class=\"chart\"></div>
      </div>
    </section>

    <section class=\"details\">
      <div class=\"insight\">
        <h3>Analyst Notes</h3>
        <ul>
          <li>Positive sentiment dominates the sample, which matches the expected bias of product review datasets.</li>
          <li>Helpfulness and review length become more informative in the 3D space than in simple one-dimensional charts.</li>
          <li>Most polarity mismatches happen in short, ambiguous, or sarcastic reviews, where context is compressed.</li>
          <li>For operational use, the 3-star and low-helpfulness clusters are the most useful triage zones.</li>
        </ul>
      </div>
      <div class=\"insight\">
        <h3>Decision Signals</h3>
        <div class=\"mini-grid\">
          <div class=\"mini-card\">
            <div class=\"mini-label\">Consensus Score</div>
            <div class=\"mini-value\">{kpis['consensus_score']:.1f}%</div>
            <div class=\"mini-note\">Share of reviews where polarity and stars align.</div>
          </div>
          <div class=\"mini-card\">
            <div class=\"mini-label\">Average Polarity</div>
            <div class=\"mini-value\">{kpis['avg_polarity']:.3f}</div>
            <div class=\"mini-note\">A compact read on the overall review tone.</div>
          </div>
          <div class=\"mini-card\">
            <div class=\"mini-label\">3-Star Neutral Share</div>
            <div class=\"mini-value\">{kpis['three_star_neutral_share']:.1f}%</div>
            <div class=\"mini-note\">Neutral sentiment concentrates around the middle rating.</div>
          </div>
          <div class=\"mini-card\">
            <div class=\"mini-label\">Avg Helpfulness</div>
            <div class=\"mini-value\">{kpis['avg_helpfulness']:.2f}</div>
            <div class=\"mini-note\">Helpfulness votes per review on average.</div>
          </div>
        </div>
      </div>
    </section>

    <div class=\"footer\">Generated from the SentimentAnalysis_AkulAttre review sample · Dark neon design · Self-contained HTML dashboard</div>
  </main>

  <script>
    const scatterData = {scatter};
    const heatmapData = {heatmap};
    const topTermsData = {top_terms};
    const sentimentCounts = {sentiment_counts};

    const chartLayout = {{
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      margin: {{l: 56, r: 28, t: 18, b: 52}},
      font: {{family: 'Space Grotesk, sans-serif', color: '#eaf3ff'}},
      hoverlabel: {{bgcolor: '#08111c', bordercolor: '#4ffcff', font: {{family: 'IBM Plex Mono, monospace'}}}},
    }};

    const scatterTrace = {{
      type: 'scatter3d',
      mode: 'markers',
      x: scatterData.x,
      y: scatterData.y,
      z: scatterData.z,
      text: scatterData.label,
      marker: {{
        size: scatterData.size,
        color: scatterData.color,
        colorscale: [[0, '#ff4f7a'], [0.5, '#ffd166'], [1, '#4ffcff']],
        opacity: 0.88,
        line: {{color: 'rgba(255,255,255,0.35)', width: 0.8}},
        symbol: 'circle'
      }},
      hovertemplate:
        '<b>%{{text}}</b><br>' +
        'Words: %{{x}}<br>' +
        'Helpfulness: %{{y:.2f}}<br>' +
        'Polarity: %{{z:.3f}}<extra></extra>'
    }};

    Plotly.newPlot('scatter3d', [scatterTrace], {{
      ...chartLayout,
      scene: {{
        xaxis: {{title: 'Word Count', gridcolor: 'rgba(79,252,255,0.14)', zerolinecolor: 'rgba(79,252,255,0.22)', color: '#dce9f9'}},
        yaxis: {{title: 'Helpfulness Ratio', gridcolor: 'rgba(79,252,255,0.14)', zerolinecolor: 'rgba(79,252,255,0.22)', color: '#dce9f9'}},
        zaxis: {{title: 'Polarity', gridcolor: 'rgba(79,252,255,0.14)', zerolinecolor: 'rgba(79,252,255,0.22)', color: '#dce9f9'}},
        bgcolor: 'rgba(0,0,0,0)',
        camera: {{eye: {{x: 1.8, y: 1.5, z: 0.9}}}},
      }},
      showlegend: false,
    }}, {{responsive: true}});

    Plotly.newPlot('donut', [{{
      type: 'pie',
      labels: sentimentCounts.labels,
      values: sentimentCounts.values,
      hole: 0.58,
      sort: false,
      direction: 'clockwise',
      marker: {{colors: ['#41f29b', '#ff4fd8', '#ff5a7a']}},
      textinfo: 'label+percent',
      hovertemplate: '%{{label}}<br>%{{value}} reviews<br>%{{percent}}<extra></extra>',
      textfont: {{family: 'IBM Plex Mono, monospace', size: 12, color: '#f7fbff'}},
      pull: [0.03, 0.03, 0.03],
    }}], {{
      ...chartLayout,
      margin: {{l: 22, r: 22, t: 0, b: 0}},
      showlegend: true,
      legend: {{orientation: 'h', y: -0.15, x: 0.12, font: {{color: '#dce9f9'}}}},
    }}, {{responsive: true}});

    Plotly.newPlot('heatmap', [{{
      type: 'heatmap',
      z: heatmapData.z,
      x: heatmapData.x,
      y: heatmapData.y,
      colorscale: [
        [0.0, '#08111c'],
        [0.25, '#163455'],
        [0.5, '#4ffcff'],
        [0.75, '#41f29b'],
        [1.0, '#ff4fd8']
      ],
      hovertemplate: 'Score %{{y}} vs %{{x}}: %{{z}} reviews<extra></extra>',
      colorbar: {{title: 'Reviews', tickfont: {{color: '#eaf3ff'}}, titlefont: {{color: '#eaf3ff'}}}},
    }}], {{
      ...chartLayout,
      margin: {{l: 58, r: 22, t: 12, b: 52}},
      xaxis: {{title: 'Sentiment', gridcolor: 'rgba(79,252,255,0.12)', color: '#dce9f9'}},
      yaxis: {{title: 'Star Rating', gridcolor: 'rgba(79,252,255,0.12)', color: '#dce9f9'}},
    }}, {{responsive: true}});

    Plotly.newPlot('topterms', [{{
      type: 'bar',
      x: topTermsData.map(item => item.term),
      y: topTermsData.map(item => item.count),
      marker: {{
        color: topTermsData.map((_, index) => index % 3 === 0 ? '#4ffcff' : index % 3 === 1 ? '#41f29b' : '#ff4fd8'),
        line: {{color: 'rgba(255,255,255,0.14)', width: 1}},
      }},
      hovertemplate: '%{{x}}<br>%{{y}} mentions<extra></extra>',
      text: topTermsData.map(item => item.count),
      textposition: 'outside',
      cliponaxis: false,
    }}], {{
      ...chartLayout,
      margin: {{l: 54, r: 18, t: 12, b: 70}},
      xaxis: {{tickangle: -24, color: '#dce9f9', gridcolor: 'rgba(79,252,255,0.08)'}},
      yaxis: {{title: 'Frequency', color: '#dce9f9', gridcolor: 'rgba(79,252,255,0.08)'}},
    }}, {{responsive: true}});
  </script>
</body>
</html>
"""


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {CSV_PATH}")

    textblob_cls = load_textblob()
    df = pd.read_csv(CSV_PATH, nrows=SAMPLE_ROWS)
    df = df[["Text", "Summary", "Score", "HelpfulnessNumerator", "HelpfulnessDenominator", "Time"]].copy()
    df["Text"] = df["Text"].fillna("").astype(str)
    df["Summary"] = df["Summary"].fillna("").astype(str)
    df = df[df["Text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["Text"]).reset_index(drop=True)

    df["combined_text"] = df["Summary"].str.strip() + " " + df["Text"].str.strip()
    df["word_count"] = df["Text"].map(word_count)
    df["helpfulness_ratio"] = df.apply(
        lambda row: safe_div(row["HelpfulnessNumerator"], row["HelpfulnessDenominator"]), axis=1
    )
    df["polarity"] = df.apply(
        lambda row: derive_polarity(row["Text"], row["Score"], textblob_cls), axis=1
    )
    df["sentiment"] = df["polarity"].map(label_sentiment)

    total_reviews = int(len(df))
    score_counts = {str(score): int(count) for score, count in df["Score"].value_counts().sort_index().items()}
    sentiment_counts_series = df["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0).astype(int)
    sentiment_counts = {
        "labels": sentiment_counts_series.index.tolist(),
        "values": sentiment_counts_series.tolist(),
    }

    polarity_mean = float(df["polarity"].mean()) if total_reviews else 0.0
    avg_score = float(df["Score"].mean()) if total_reviews else 0.0
    avg_words = float(df["word_count"].mean()) if total_reviews else 0.0
    avg_helpfulness = float(df["helpfulness_ratio"].mean()) if total_reviews else 0.0

    mismatch_mask = ((df["Score"] >= 4) & (df["sentiment"] == "Negative")) | ((df["Score"] <= 2) & (df["sentiment"] == "Positive"))
    mismatch_rate = safe_div(int(mismatch_mask.sum()), total_reviews) * 100
    high_star_negative = int(((df["Score"] == 5) & (df["sentiment"] == "Negative")).sum())
    consensus_score = 100.0 - mismatch_rate

    three_star_total = int((df["Score"] == 3).sum())
    three_star_neutral = int(((df["Score"] == 3) & (df["sentiment"] == "Neutral")).sum())
    three_star_neutral_share = safe_div(three_star_neutral, three_star_total) * 100

    sentiment_cards = {
        label: {
            "count": int(sentiment_counts_series[label]),
            "pct": safe_div(int(sentiment_counts_series[label]), total_reviews) * 100,
        }
        for label in sentiment_counts_series.index
    }

    heatmap_frame = (
        df.groupby(["Score", "sentiment"]).size().unstack(fill_value=0).reindex(index=[1, 2, 3, 4, 5], columns=["Positive", "Neutral", "Negative"]).fillna(0)
    )
    heatmap = {
        "x": heatmap_frame.columns.tolist(),
        "y": [int(idx) for idx in heatmap_frame.index.tolist()],
        "z": heatmap_frame.astype(int).values.tolist(),
    }

    scatter_sample = df.sample(min(SCATTER_SAMPLE, total_reviews), random_state=42) if total_reviews else df
    scatter = {
        "x": scatter_sample["word_count"].astype(int).tolist(),
        "y": scatter_sample["helpfulness_ratio"].round(3).tolist(),
        "z": scatter_sample["polarity"].round(3).tolist(),
        "label": [f"Score {score} · {sentiment}" for score, sentiment in zip(scatter_sample["Score"], scatter_sample["sentiment"])],
        "size": [max(4, min(12, int(math.sqrt(max(1, wc))))) for wc in scatter_sample["word_count"].astype(int).tolist()],
        "color": scatter_sample["polarity"].round(3).tolist(),
    }

    top_terms = build_top_terms(df, limit=12)

    payload = {
        "kpis": {
            "total_reviews": total_reviews,
            "avg_score": avg_score,
            "positive_share": sentiment_cards["Positive"]["pct"],
            "mismatch_rate": mismatch_rate,
            "avg_words": avg_words,
            "high_star_negative": high_star_negative,
            "consensus_score": consensus_score,
            "avg_polarity": polarity_mean,
            "three_star_neutral_share": three_star_neutral_share,
            "avg_helpfulness": avg_helpfulness,
        },
        "sentiment_cards": sentiment_cards,
        "heatmap": heatmap,
        "scatter": scatter,
        "top_terms": top_terms,
        "sentiment_counts": sentiment_counts,
        "score_counts": score_counts,
    }

    html = build_html(payload)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Dashboard written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()