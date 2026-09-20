# Reproduced research results

Run timestamp: 20 September 2026. Dataset: NIFTY 50 daily observations obtained by the configured Yahoo research adapter, 24 September 2018 through 18 September 2026. The fixed primary evaluation is a chronological 70/15/15 split. These results describe one provider snapshot and are not evidence of future profitability.

## Direction classification: untouched test segment

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.481 | 0.472 | 0.286 | 0.356 | 0.465 |
| Random Forest | 0.464 | 0.472 | 0.565 | 0.514 | 0.450 |
| XGBoost | 0.481 | 0.488 | 0.701 | 0.575 | 0.481 |

None of the tested classifiers demonstrated useful discrimination on this holdout: all ROC-AUC values were below 0.50. The comparatively higher XGBoost recall does not offset its weak discrimination and should not be read as predictive success.

## Context ablation

| Feature set (Logistic Regression) | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Technical | 0.481 | 0.356 | 0.465 |
| Technical + causal regime | 0.502 | 0.447 | 0.466 |
| Technical + regime + sentiment | Not run | Not run | Not run |

The causal regime feature modestly increased accuracy and F1 but ROC-AUC remained below chance. Each state used here was fitted only on earlier observations; the original full-sample state assignment was rejected during audit as look-ahead leakage. The sentiment stage was intentionally skipped because no news API credential or time-aligned articles were available; the experiment runner does not invent or zero-fill missing news context.

## Regime separation

| Algorithm | Silhouette | Calinski–Harabasz |
|---|---:|---:|
| K-Means | 0.290 | 719.54 |
| Gaussian Mixture | 0.172 | 418.00 |
| Gaussian HMM | 0.174 | 429.52 |

K-Means produced the strongest geometric separation in this run. HMM remains useful for transition/persistence analysis, not because it had the best separation score.

## NIFTY 50 walk-forward backtest

Configured assumptions: daily close-derived signals executed next period, 10 bps transaction cost plus 5 bps slippage on position changes. Period uses all available observations. The ML strategies are long/cash at a 0.55 probability threshold.

| Strategy | Cumulative return | CAGR | Sharpe | Max drawdown | Position changes |
|---|---:|---:|---:|---:|---:|
| Buy and hold | 101.3% | 9.27% | 0.600 | -38.4% | 1 |
| SMA baseline | 23.3% | 2.69% | 0.263 | -36.4% | 111 |
| Walk-forward supervised ML | 9.5% | 1.16% | 0.221 | -8.7% | 130 |
| Regime-aware ML | 21.7% | 2.52% | 0.469 | -10.5% | 130 |
| Regime + sentiment hybrid | Not run | Not run | Not run | Not run | Not run |

Buy and hold dominated return and Sharpe in this sample. The supervised and causal regime-aware ML strategies reduced maximum drawdown materially but delivered much less return. The unavailable hybrid is exposed as an explicit API/UI error until genuine time-aligned news is ingested.

Reproduce locally using the commands in the README; provider revisions or a later end date may change every number.
