# AI-Based Stock Market Intelligence and Prediction Platform

A four-page application for exploring Indian stocks and understanding what its saved models estimate from the latest available market history. It is a research and education project, not a trading system or investment advice. The direction model's measured real-market performance is close to chance; the application displays this limitation alongside its outlook.

## Overview

The app follows one simple path: check the market in **Overview**, find a company in **Market Explore**, examine its history and same-sector peers in **Stock Intelligence**, then request an experimental outlook in **AI Predictions**. The same selected stock carries between pages. Market observations come from a real provider; model estimates are labelled separately.

## Features

### Market Overview

NIFTY 50, NIFTY Bank and SENSEX closing values; latest-session changes; tracked-stock breadth, volume, gainers, losers and an interactive index chart.

### Market Explore

Search, sort, select a stock, filter by one of nine configured sectors, refresh provider data on demand, and inspect actual price and volume history. The table shows its observation date and an Analyze link.

### Stock Intelligence

One-stock price, daily change, volume, approximate 52-week range, price chart and plain-language technical summary. Additional indicators are collapsed by default. Similar stocks are compared **only within the selected stock's sector**; sectors with too few tracked companies use nearest same-sector peers.

### AI Predictions

A saved real-market classifier supplies an experimental UP/DOWN direction, model probability and plain-language SHAP reasons. Saved Lasso or Ridge models supply a future price and daily volatility estimate. The app selects models from validation results; users do not choose an algorithm. The real direction classifier has **no demonstrated predictive advantage** on its held-out data. Its result must not be used as an investment recommendation.

## System Architecture

```text
Yahoo Finance daily OHLCV → validation → SQLite (default) / PostgreSQL (optional)
                                      │
                                      ├→ latest prices and technical features → four-page UI
                                      ├→ same-sector behavioral clustering
                                      └→ offline training → saved models
                                           ├→ Logistic Regression / XGBoost direction
                                           └→ Lasso / Ridge price and volatility
                                      ↓
                     FastAPI REST + WebSocket → HTML/CSS/JavaScript + Chart.js
```

Training runs during setup or by an explicit command. Clicking Analyze loads saved artifacts and computes SHAP explanations; it does not retrain. A separate controlled synthetic dataset exists for a documented research experiment and is **never** used as the selected real stock's prediction input.

## Technology Stack

- **Python 3.11**, FastAPI, Uvicorn, SQLAlchemy and Alembic
- pandas, NumPy, scikit-learn, XGBoost, SHAP and joblib
- **SQLite by default**; PostgreSQL via `psycopg` is optional
- Yahoo Finance through `yfinance` for latest available daily research data
- HTML5, CSS3, vanilla JavaScript and Chart.js for the frontend

Dependencies are declared in [pyproject.toml](pyproject.toml). There is no separate `requirements.txt`; `pip install -e ".[dev]"` installs the application and test tools from that single dependency list.

## Requirements

- Windows 10/11, 64-bit **Python 3.11** (tested with 3.11.5) and Git
- Internet access for installation and Yahoo Finance ingestion; provider access may vary by region or time
- VS Code is recommended, but not required
- No database server or API key is needed for the default SQLite setup
- PowerShell and the Python launcher (`py`) or `python` on `PATH`

## Quick Start on Windows

Open PowerShell in the folder where you want the project:

```powershell
git clone https://github.com/sohamkamerkar007/ai-stock-market-intelligence.git
cd ai-stock-market-intelligence
.\setup.ps1
.\run.ps1
```

Open **http://127.0.0.1:8000**. To open the code in VS Code, run `code .` from the repository (or use **File → Open Folder**). Setup may take several minutes: it creates `.venv`, installs dependencies, creates `.env` only if absent, runs the database migration, fetches roughly eight years of actual daily history, and trains the saved real-market direction, price and volatility models. No current-machine database or model file is required by a clean clone. The downloaded database, model artifacts and local reports stay on that computer and are Git-ignored.

If PowerShell blocks local scripts, run the same commands through a one-process policy override:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1
```

The default app port is 8000. Use `.\run.ps1 -Port 8001` if that port is occupied. Stop the server with **Ctrl+C**.

### Manual equivalent

The scripts use these commands from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m scripts.bootstrap --years 8 --strict
.\.venv\Scripts\python.exe -m scripts.train_real_direction
.\.venv\Scripts\python.exe -m scripts.train_final_stock_models
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

You can activate the environment with `.\.venv\Scripts\Activate.ps1`, but activation is optional because every command above names its Python executable. If data-provider access is temporarily unavailable, `setup.ps1 -SkipData` initializes the app and database without downloading prices; rerun plain `.\setup.ps1` later to populate all pages and train models. `-SkipModels` downloads data but leaves AI Predictions unavailable until you train the models.

## Environment Configuration

Setup copies [.env.example](.env.example) to `.env` only when `.env` does not already exist. **Never commit `.env`**. The example uses `DATABASE_URL=sqlite:///./market_intelligence.db`; this creates a local file automatically when Alembic runs. There is no secret to fill in for the four-page app. `NEWSDATA_API_KEY` is optional and used only by legacy news research commands. `MODEL_DIR`, `CACHE_DIR`, `LOG_LEVEL`, `CORS_ORIGINS` and research settings have safe defaults in the example. Set a strong, private database password only if you choose PostgreSQL.

The default SQLite database needs no installation. Setup runs `python -m alembic upgrade head`, and `scripts.bootstrap` seeds the tracked index and stock universe. To use PostgreSQL, configure `DATABASE_URL` and matching `POSTGRES_*` values in your untracked `.env`, start a local server (or `docker compose up -d db`), then rerun the migration and bootstrap commands. Docker is optional; it is not needed for the recommended Windows path.

## Market Data and Refresh

Yahoo Finance supplies **latest available daily** OHLCV research bars. This is not an exchange-grade live or tick feed. Charts use actual trading dates and prices. The latest session may be yesterday or older on weekends, holidays, provider outages or before a daily bar is published; a 1D range contains one daily observation. On-demand refresh validates bars, updates the latest session, preserves earlier rows and prevents duplicates. If the provider fails, the app keeps the last verified observations and displays their date instead of inventing a current quote. Run `.\setup.ps1` again to repair an incomplete first download, or use **Get latest data** inside the app for later refreshes.

## Model Setup and Interpretation

- **Classification:** Logistic Regression and XGBoost are trained on actual tracked-stock history for 1, 3 and 5 trading-session direction. The saved classifier is selected by validation F1, then validation accuracy. Inference uses the same feature schema, imputer and scaler as training. SHAP contributions become short explanations. The model probability is not a calibrated success rate.
- **Regression:** Lasso and Ridge estimate future return and daily volatility from real stock history. Validation MAE chooses the saved model separately for each target; future return is converted to an estimated rupee price using the latest actual close.
- **Clustering:** standardized behavioral features are compared within the selected stock's sector. Groups describe historical similarity; they are not predictions.
- **Controlled synthetic research:** an optional, fixed-seed research panel trains separate classifiers and produces about 81% synthetic test accuracy. That number is **not** real-stock accuracy. The synthetic experiment does not meet the requested 80% baseline threshold; its training-majority constant baseline scores 43.68% on its shifted test period.

All evaluations use chronological splits and purge labels that cross split boundaries. The real direction model's balanced accuracy is approximately 50% on held-out data, so it has no demonstrated forecasting edge. Read the [final model report](docs/FINAL_MODEL_REPORT.md) and [methodology](docs/FINAL_PREDICTIONS.md) before interpreting a result. Historical price-level R² does not establish a profitable forecast.

Model binaries are generated locally under `models/final/` and are **not committed**. Runtime training reports are written there too. The repository includes measured snapshot reports in `docs/` for review. To reproduce the optional controlled experiment, run:

```powershell
.\.venv\Scripts\python.exe -m scripts.generate_final_research_dataset
.\.venv\Scripts\python.exe -m scripts.train_final_classifiers
```

That optional command downloads no third-party synthetic file: the dataset is generated locally from the documented fixed-seed generator. It can update the committed research snapshot report, so review Git status afterward.

## Run the Application

```powershell
.\run.ps1
```

The app and API documentation are at **http://127.0.0.1:8000** and **http://127.0.0.1:8000/docs**. The frontend is served by FastAPI; no Node installation or separate frontend server is needed. The four visible pages are Overview, Market Explore, Stock Intelligence and AI Predictions.

## Tests and Troubleshooting

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check backend src scripts tests
```

- **`py -3.11` missing:** install 64-bit Python 3.11 and enable its launcher, or use `python` if it is Python 3.11.
- **Provider unavailable or rate limited:** retry setup later. Existing verified observations stay in the database; do not substitute fabricated data.
- **AI artifact missing:** rerun plain `.\setup.ps1` after market history is present. The setup script retrains and saves required models.
- **Port 8000 occupied:** use `.\run.ps1 -Port 8001` and open the corresponding address.
- **Custom PostgreSQL fails:** check the private `.env` connection string and that the server is running; switch back to the SQLite URL for the simplest local setup.

Further details: [API](docs/API.md), [data sources](docs/DATA_SOURCES.md), [architecture](docs/ARCHITECTURE.md), [limitations](docs/LIMITATIONS.md) and [research](docs/RESEARCH.md).
