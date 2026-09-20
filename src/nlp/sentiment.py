import re
from dataclasses import dataclass
from datetime import time
from zoneinfo import ZoneInfo

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass
class SentimentResult:
    label: str
    score: float
    confidence: float
    model: str


class VaderFinancialBaseline:
    name = "vader_financial_baseline"

    def __init__(self) -> None:
        self.analyzer = SentimentIntensityAnalyzer()
        self.analyzer.lexicon.update(
            {
                "upgrade": 2.0,
                "downgrade": -2.0,
                "outperform": 2.2,
                "underperform": -2.2,
                "profit": 1.5,
                "loss": -1.5,
                "fraud": -3.0,
                "default": -2.5,
            }
        )

    def analyze(self, text: str) -> SentimentResult:
        score = float(self.analyzer.polarity_scores(text or "")["compound"])
        label = "positive" if score >= 0.05 else "negative" if score <= -0.05 else "neutral"
        return SentimentResult(
            label, score, min(1.0, abs(score) + (0.2 if label == "neutral" else 0.0)), self.name
        )


def associate_symbols(text: str, aliases: dict[str, list[str]]) -> list[str]:
    lowered = (text or "").lower()
    return sorted(
        symbol
        for symbol, names in aliases.items()
        if any(re.search(rf"\b{re.escape(name.lower())}\b", lowered) for name in names)
    )


def daily_sentiment(articles: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Aggregate only articles known by cutoff; protects feature construction from future news."""
    data = articles.copy()
    data["published_at"] = pd.to_datetime(data["published_at"], utc=True)
    data = data[data["published_at"] <= pd.Timestamp(cutoff)]
    if data.empty:
        return pd.DataFrame(
            columns=["date", "symbol", "sentiment_mean", "news_volume", "sentiment_momentum"]
        )
    data["date"] = data["published_at"].map(effective_market_date)
    daily = data.groupby(["date", "symbol"], as_index=False).agg(
        sentiment_mean=("sentiment_score", "mean"), news_volume=("sentiment_score", "size")
    )
    daily["sentiment_momentum"] = daily.groupby("symbol")["sentiment_mean"].diff()
    return daily


def effective_market_date(published_at: object) -> pd.Timestamp:
    """Return the first market date that can legitimately use an article.

    Articles published after the 15:30 IST close are moved to the following
    calendar day. A later as-of merge against observed sessions naturally
    carries weekends and exchange holidays forward.
    """
    stamp = pd.Timestamp(published_at)
    if stamp.tzinfo is None:
        raise ValueError("News publication timestamps must be timezone-aware")
    local = stamp.tz_convert(ZoneInfo("Asia/Kolkata"))
    effective = local.normalize()
    if local.time() > time(15, 30):
        effective += pd.Timedelta(days=1)
    return effective.tz_convert("UTC")
