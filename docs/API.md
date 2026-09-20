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
| GET | `/api/v1/anomalies` | Statistical anomalies |
| GET | `/api/v1/research/models` | Actual experiment runs |
| POST | `/api/v1/backtests` | Run configured baseline backtest |
| WS | `/ws/market` | Freshness-aware persisted market snapshots |

Errors use FastAPI's structured `{"detail": ...}` response. Unknown assets return 404; insufficient analysis history returns 409.

