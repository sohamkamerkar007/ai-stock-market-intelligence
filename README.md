# Bharat Market Intelligence

An end-to-end research platform for Indian equities: reproducible market-data ingestion, leakage-aware features, supervised direction models, unsupervised market environments, contextual ablation studies, news sentiment, unusual-activity detection, SHAP explanations, time-aware historical tests, a FastAPI API/WebSocket layer, and a framework-free financial-intelligence UI. The primary interface uses plain language while preserving technical detail through progressive disclosure.

> **Research use only.** Outputs are uncertain statistical/model estimates, not investment advice, an execution system, or a claim of market-beating performance. The default data path is end-of-day research data; the application never labels it real-time.

## Architecture

```text
Yahoo research EOD       Licensed quote adapter path  NewsData.io (optional)
        |                         |                           |
        +----------- provider adapters / validation ---------+
                                  |
                      PostgreSQL + Alembic schema
                                  |
        features -> regimes -> supervised/hybrid -> anomalies -> SHAP
                                  |
                         walk-forward backtests
                                  |
             FastAPI REST + freshness-aware WebSocket
                                  |
                   HTML/CSS/vanilla JS research UI
```

Detailed design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Capabilities

- 23-asset starting universe: NIFTY 50, BANK NIFTY, SENSEX and 20 liquid equities across major sectors.
- Idempotent, incremental OHLCV ingestion with validation, retries, timestamps, failure records and provider isolation.
- One reusable close-of-day feature pipeline; next-day targets are shifted forward and never imputed.
- Logistic Regression, Random Forest and XGBoost with chronological train/validation/test splits.
- K-Means, Gaussian Mixture and Hidden Markov regimes, described from observed state statistics after fitting.
- VADER financial baseline; optional FinBERT dependencies; news is aggregated only up to the prediction cutoff.
- Isolation Forest + Local Outlier Factor ensemble, distinct from prediction.
- SHAP attribution and a constrained human-language translation layer.
- Cost-aware next-period backtests and ten functioning application views.

## Windows setup

Prerequisites: Git, Python 3.11, and either PostgreSQL 16 or Docker Desktop.

```powershell
git clone https://github.com/sohamkamerkar007/ai-stock-market-intelligence.git
cd ai-stock-market-intelligence
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

For the simplest local evaluation, leave the default SQLite URL created by the script. For the required production database, start PostgreSQL and use the `.env.example` URL:

```powershell
Copy-Item .env.example .env
docker compose up -d db
.\.venv\Scripts\python.exe -m alembic upgrade head
```

Set a strong local `POSTGRES_PASSWORD` in `.env` and keep the matching password in `DATABASE_URL`; Compose contains no embedded database credential.

## Acquire data and run research

```powershell
# Eight years of legitimate provider data; caches are not committed
.\.venv\Scripts\python.exe -m scripts.bootstrap --years 8

# Bounded experiment matrix and published production selection (actual outputs only)
.\.venv\Scripts\python.exe -m scripts.run_research_matrix --publish
.\.venv\Scripts\python.exe -m scripts.train_regimes --symbol NIFTY50 --algorithm hmm
.\.venv\Scripts\python.exe -m scripts.detect_anomalies
.\.venv\Scripts\python.exe -m scripts.generate_predictions
.\.venv\Scripts\python.exe -m scripts.audit_research_results

# Separate controlled synthetic ML-research experiment (not real-market accuracy)
.\.venv\Scripts\python.exe -m scripts.generate_synthetic_market_data

# Optional, requires NEWSDATA_API_KEY
.\.venv\Scripts\python.exe -m scripts.ingest_news
```

Provider terms and exact limitations are in [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md). Do not schedule aggressive downloads; incremental ingestion requests only missing dates.

## Run

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000
```

Open `http://localhost:8000`. API documentation is at `http://localhost:8000/docs`.

Containerized full stack:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

## Test and lint

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check backend src scripts tests
```

## Environment variables

| Variable | Required | Purpose |
|---|---:|---|
| `DATABASE_URL` | Yes | PostgreSQL SQLAlchemy URL; SQLite is supported for local tests |
| `NEWSDATA_API_KEY` | Optional | News ingestion |
| `MODEL_DIR`, `CACHE_DIR` | No | Local artifact/cache locations |
| `TRANSACTION_COST_BPS`, `SLIPPAGE_BPS` | No | Reproducible backtest assumptions |
| `CORS_ORIGINS`, `LOG_LEVEL`, `APP_ENV` | No | API runtime settings |

## Real-market baseline snapshot

The bounded experiment matrix deliberately records weak as well as strong outcomes. On the included retrospective NIFTY holdout, the pre-holdout-selected ExtraTrees technical-plus-market model produced 50.39% accuracy, 50.28% balanced accuracy, 0.5646 F1, 0.5326 ROC-AUC and 0.0059 MCC (258 sessions). The majority benchmark was also 50.39%; this is not evidence of useful one-day predictive skill. The full diagnosis, protocol, legacy comparison and limitations are in [ML performance diagnosis](docs/ML_PERFORMANCE_DIAGNOSIS.md).

The separate controlled synthetic research experiment is designed to demonstrate the supervised ML workflow on learnable market-like data. Its 70%+ result is never presented as real-market accuracy; see [Controlled synthetic research dataset](docs/SYNTHETIC_RESEARCH_DATASET.md).

After data ingestion and training, actual experiment outputs appear in PostgreSQL, the Research Lab, model-sidecar JSON files, and:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_models
```

This makes the experiment date, provider history, dependency versions and local run explicit rather than presenting stale numbers as universal results. Run `scripts.run_research_matrix --publish` to recreate the current research view.

## Documentation

- [Data sources](docs/DATA_SOURCES.md) · [ML methodology](docs/ML_METHODOLOGY.md) · [Backtesting](docs/BACKTESTING.md) · [Controlled synthetic research dataset](docs/SYNTHETIC_RESEARCH_DATASET.md)
- [API](docs/API.md) · [Research design](docs/RESEARCH.md) · [Reproduced results](docs/RESULTS.md) · [Limitations](docs/LIMITATIONS.md)
- [Data dictionary](docs/DATA_DICTIONARY.md) · [Architecture](docs/ARCHITECTURE.md)
