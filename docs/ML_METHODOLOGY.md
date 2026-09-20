# ML methodology

## Observation and target

Features are calculated using OHLCV available at the close of trading day *t*. The primary target is `close[t+1] / close[t] - 1 > 0`. The final unknown target is null and excluded, never converted to DOWN. A production decision made before the close must shift or replace close-derived features accordingly.

## Features

Returns, log returns, SMA/EMA ratios, MACD, RSI, momentum, annualized rolling volatility, ATR, Bollinger width, volume change, relative volume and drawdown are defined once in `src/features/pipeline.py`. Rolling calculations are backward-looking. Optional index/sector/news inputs must be aligned backward/as-of to the prediction timestamp.

## Evaluation

The standard split is chronological 70% train, 15% validation and 15% untouched test. No time-series shuffle is used. Logistic Regression is the interpretable baseline; Random Forest and XGBoost add nonlinear baselines. Accuracy, precision, recall, F1 and ROC-AUC are recorded. The walk-forward helper retrains only on preceding observations.

## Regimes

K-Means, Gaussian Mixture and Gaussian HMM use standardized return, volatility, momentum, volume and drawdown features. Numeric states are discovered first. Labels are generated afterward from each state's observed statistics. Silhouette and Calinski–Harabasz scores compare separation; HMM transition matrices describe persistence. Economic plausibility and stability across windows are required beyond a single clustering score.

## Contextual ablation

`run_hybrid_experiment.py` compares the same classifier and chronological protocol across technical, technical+regime, and technical+regime+sentiment sets. Missing context causes a skipped experiment, not imputation with invented information. A contextual model is not assumed to outperform.

## Reproducibility

Random seeds are fixed at 42, model artifacts use unique run IDs, and metadata records date coverage, features, parameters and metrics. Full deterministic reproduction also requires pinning the resolved environment and preserving the provider snapshot under permitted terms.

