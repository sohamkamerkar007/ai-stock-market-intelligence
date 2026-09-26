# Controlled synthetic prediction-research dataset

## Purpose and boundary

This dataset is an explicitly synthetic, controlled market-like panel used to demonstrate the platform's supervised-learning methodology when real next-day Indian-market direction has too little stable OHLCV-only signal. It does **not** represent Indian exchange data and its scores do **not** estimate, validate, or imply real-market forecasting accuracy.

Real provider data remains the sole source for the Overview, Market, Stock Intelligence, Market Environment, News Intelligence, Unusual Activity, and Historical Performance pages. The AI Predictions page is labelled as a separate real-market baseline; it does not borrow the synthetic experiment's accuracy.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m scripts.generate_synthetic_market_data
```

The default fixed seed is `20260926`. The command writes the ignored data file to `data/processed/synthetic_market_research.csv.gz`, the ignored selected pipeline to `models/synthetic-research-selected.joblib`, and the small, versioned measurement record to `docs/synthetic_research_results.json`.

## Generator

The default run contains 24 synthetic assets (three in each of IT, Banking, Financial Services, Energy, FMCG, Pharma, Auto, and Telecom) over 2,200 business sessions from 2018-01-01 to 2026-06-05. It produces OHLCV and adjusted-close paths from a persistent four-state market environment: bearish, bullish, range-bound, and high-volatility.

At each session, the next simulated return direction is sampled from a probability affected by current/past observable market environment, recent asset trend, sector and broad-market return persistence, activity persistence, asset heterogeneity, and an independent noise term. The latent innovation itself is never saved as a model feature. The result is deliberately learnable but not deterministic; it is not a shortcut such as `target = feature > threshold`.

Features are calculated at the current session close only: return/log-return, SMA/EMA trend, RSI/MACD/momentum, ATR/Bollinger/rolling volatility, volume statistics, broad-market and sector context, and current environment fields. For a horizon `h`, `target_h` is computed only from the synthetic close at `t+h`, with `target_available_at_h` stored and used to purge split boundaries.

The default label is UP when the forward synthetic close-to-close return exceeds +0.10%; otherwise it is DOWN. One-, three-, and five-session targets are compared.

## Evaluation protocol

Rows are ordered by date and partitioned by date: first 70% train, next 15% validation, and latest 15% final test. Labels that would become known in the following period are removed from train/validation boundary rows. Model parameters, target horizon, and feature group are chosen from validation balanced accuracy and then ROC-AUC only. The selected configuration is refit on the eligible train-plus-validation period and evaluated once on the final test period.

The experiment compares Logistic Regression, Random Forest, XGBoost, and Histogram Gradient Boosting. Feature groups are technical only; technical + market; technical + market + environment; and technical + market + environment + volume. The saved report includes class prevalence, majority baseline, full metrics, confusion matrix, per-asset results, per-environment results, feature importance, and exact split dates.

## Limitations

- Controlled relationships are intentionally present, so the resulting 70%+ score is a methodology demonstration rather than external validity.
- Synthetic regimes and sectors do not capture corporate actions, execution, point-in-time constituents, or the unobserved information flow of a real exchange.
- The experiment must never be used to infer expected returns, trading profitability, or the accuracy of the real-market baseline model.
