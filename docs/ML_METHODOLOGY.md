# ML methodology

## Observation and target

Features are calculated using OHLCV available at the close of trading day *t*. The primary target is `close[t+1] / close[t] - 1 > 0`. The final unknown target is null and excluded, never converted to DOWN. A production decision made before the close must shift or replace close-derived features accordingly.

## Features

Returns, log returns, SMA/EMA ratios, MACD, RSI, momentum, annualized rolling volatility, ATR, Bollinger width, volume change, relative volume and drawdown are defined once in `src/features/pipeline.py`. Rolling calculations are backward-looking. Optional index/sector/news inputs must be aligned backward/as-of to the prediction timestamp.

## Evaluation

The current research matrix uses three expanding development folds with horizon-length purges, then a sealed final 15% retrospective holdout. Configuration selection uses development balanced accuracy, with Brier score as tie-breaker. Horizons 1/3/5 are compared, with one-day remaining primary. Preprocessing and calibration are fitted only on earlier observations. Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, MCC, Brier score, confusion matrices and fold variation are recorded. The old holdout was already inspected and must not be described as an untouched prospective test. See `ML_PERFORMANCE_DIAGNOSIS.md` for all results and limitations.

## Regimes

K-Means, Gaussian Mixture and Gaussian HMM use standardized return, volatility, momentum, volume and drawdown features. Numeric states are discovered first. Labels are generated afterward from each state's observed statistics. Silhouette, Calinski–Harabasz and repeat-fit adjusted Rand scores assess separation and basic stability; HMM transition matrices describe persistence. Persisted full-sample timelines are descriptive only. Predictive experiments and backtests use expanding-window GMM assignments fitted strictly on preceding observations.

## Contextual ablation

`run_hybrid_experiment.py` compares the same classifier and chronological protocol across technical, technical+regime, and technical+regime+sentiment sets. Missing context causes a skipped experiment, not imputation with invented information. A contextual model is not assumed to outperform.

## Reproducibility

Random seeds are fixed at 42, model artifacts use unique run IDs, and metadata records date coverage, features, parameters and metrics. Full deterministic reproduction also requires pinning the resolved environment and preserving the provider snapshot under permitted terms.
