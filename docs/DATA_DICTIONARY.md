# Data dictionary

| Table | Key content |
|---|---|
| `assets` | canonical symbol, provider symbol, name, exchange, sector, asset type |
| `asset_prices` | unique asset/timestamp/interval OHLCV observation, adjusted close, provider |
| `technical_features` | versioned JSON feature vector at an asset timestamp |
| `market_regimes` | algorithm, numeric state, post-hoc description and probability |
| `model_runs` | experiment ID, task, learner, feature set, dates, parameters, actual metrics, artifact |
| `predictions` | target date, direction, `probability_up`, expected return and regime context |
| `prediction_explanations` | feature contributions and constrained natural-language summary |
| `news_articles` | provider ID, headline/description, source, URL, publication time and symbols |
| `news_sentiment` | model, class, signed score and confidence |
| `anomalies` | detector, type, score, severity and observed context |
| `backtests` | strategy, period, assumptions, metrics and equity curve |
| `data_ingestion_runs` | provider, start/end, status, received/written counts and failures |

Timestamps are normalized to timezone-aware UTC when ingested. Application display uses the browser locale; provider publication timestamps remain traceable.

