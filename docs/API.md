# API

Interactive OpenAPI documentation is available at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Database/API health |
| GET | `/api/v1/assets` | Configured asset universe |
| GET | `/api/v1/overview` | Indices, breadth, movers, latest context |
| GET | `/api/v1/assets/{symbol}/prices` | Historical OHLCV |
| GET | `/api/v1/assets/{symbol}/features` | Calculated technical series |
| GET | `/api/v1/regimes` | Persisted regime timeline |
| GET | `/api/v1/predictions` | Persisted probabilistic predictions |
| GET | `/api/v1/predictions/{id}/explanation` | Grounded model attribution |
| GET | `/api/v1/news` | News and sentiment |
| POST | `/api/v1/news/analyze` | Non-persistent sentiment and context analysis for user-supplied financial text |
| GET | `/api/v1/anomalies` | Statistical anomalies |
| GET | `/api/v1/research/models` | Actual experiment runs |
| GET | `/api/v1/research/experiments` | Complete measured matrix, fold results, holdout diagnostics and asset comparisons |
| GET | `/api/v1/research/synthetic` | Explicitly labelled controlled synthetic-data ML experiment and final-test measurements |
| GET | `/api/v1/regimes/summary` | Named categorical periods, observed statistics and transitions |
| POST | `/api/v1/backtests` | Run configured baseline backtest |
| WS | `/ws/market` | Freshness-aware persisted market snapshots |

Errors use FastAPI's structured `{"detail": ...}` response. Unknown assets return 404; insufficient analysis history returns 409.
