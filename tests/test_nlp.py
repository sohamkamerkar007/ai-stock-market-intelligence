import pandas as pd

from src.nlp.sentiment import VaderFinancialBaseline, daily_sentiment, effective_market_date


def test_sentiment_and_cutoff():
    scorer = VaderFinancialBaseline()
    assert scorer.analyze("profit upgrade and strong growth").score > 0
    rows = pd.DataFrame(
        [
            {"published_at": "2025-01-01T08:00:00Z", "symbol": "TCS", "sentiment_score": 0.5},
            {"published_at": "2025-01-03T08:00:00Z", "symbol": "TCS", "sentiment_score": -1},
        ]
    )
    daily = daily_sentiment(rows, pd.Timestamp("2025-01-02", tz="UTC"))
    assert len(daily) == 1
    assert daily.sentiment_mean.iloc[0] == 0.5


def test_post_close_news_moves_to_next_market_date():
    before_close = effective_market_date("2025-01-03T09:59:00Z")
    after_close = effective_market_date("2025-01-03T10:01:00Z")
    assert before_close == pd.Timestamp("2025-01-02T18:30:00Z")
    assert after_close == pd.Timestamp("2025-01-03T18:30:00Z")
