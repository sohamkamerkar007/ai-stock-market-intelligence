# Final model report (2 October 2026)

The real-market and controlled synthetic evaluations are separate. The app uses only actual market history for stock inference. These measured results do not establish a profitable forecast.

## Real-market direction classification

The target is UP when the close after 1, 3, or 5 trading sessions is above the current close; otherwise it is DOWN. Current-close features include returns, moving-average distance, momentum, volatility, volume, NIFTY 50 return, same-sector average return, and relative performance. Each horizon uses a chronological 65/15/20 train/validation/test split. Labels crossing split boundaries are purged. Logistic Regression and XGBoost are selected by validation balanced accuracy, then F1; the held-out test does not select the model or horizon. The default 1-session horizon had the best selected-model validation balanced accuracy among the three.

| Horizon | Selected model | Test accuracy | Balanced accuracy | F1 | ROC-AUC | Training-majority baseline |
|---|---|---:|---:|---:|---:|---:|
| 1 session | XGBoost | 50.39% | 50.47% | 51.33% | 0.512 | 48.75% |
| 3 sessions | Logistic Regression | 51.26% | 51.29% | 51.35% | 0.518 | 48.87% |
| 5 sessions | Logistic Regression | 50.93% | 51.02% | 54.12% | 0.514 | 49.39% |

For the validation-selected 1-session default, XGBoost test accuracy is **50.39%** and Logistic Regression test accuracy is **50.47%**. XGBoost precision is **49.19%**, recall **53.65%**, and confusion matrix (DOWN/UP) `[[4064, 4530], [3789, 4386]]`. The test interval is 20 February 2025 through 30 September 2026. [The complete report](real_direction_results.json) includes both learners, all horizons, split dates, and confusion matrices.

The original approximately 48.7% result arose from a model-selection rule favoring F1 in a shifting class distribution. XGBoost predicted UP often, producing high recall and F1 while balanced accuracy stayed near 50%. New backward-looking features, market and sector context, and selection by validation balanced accuracy modestly changed the result. The real-market classifier still has **no demonstrated predictive advantage**.

## Controlled Synthetic Research Evaluation

The separate controlled 3-session panel has 15,768 held-out rows. Validation selected Logistic Regression, which reached **81.32%** test accuracy, **81.04%** balanced accuracy, **78.65%** F1, and **0.890** ROC-AUC. XGBoost reached 81.37% test accuracy but did not win the prespecified validation comparison. The training-majority constant baseline scored 43.68% test accuracy. This demonstrates performance only on the intentionally learnable simulated panel; it is not real-market accuracy. [Full controlled results](final_classification_results.json) include the evaluation protocol and class distribution.

## Price, volatility, and clustering

Lasso and Ridge estimates are trained per stock and horizon on actual daily history. Validation MAE selects each target's model. [The regression report](final_stock_regression_results.json) contains the latest held-out measurements for all 43 tracked stocks and three horizons. Price estimates and expected daily volatility are model outputs, not guarantees.

The authoritative universe now contains 43 tracked stocks in nine approved sectors. Every added Yahoo Finance symbol was checked for available provider history and then ingested. Clustering standardizes six backward-looking features within each sector. A peer group is withheld when fewer than four stocks have sufficient history; a two-cluster partition is used only when every resulting group has at least three members. API checks for 12 selected stocks, including five banks, returned only same-sector peers.

## Explanation and confidence rules

SHAP contributions come from the saved selected direction model and the latest real feature row. The UI translates the leading positive and negative contributions into market-signal language. A missing explanation is stated as unavailable. Model confidence is `max(P(UP), 1 − P(UP))` and is **not calibrated success probability**. The UI calls values below 60% “Mixed signals,” 60% to below 75% “Moderate model preference,” and 75% or above “Strong model preference.” Expected daily volatility is labelled lower below 1%, moderate from 1% to below 2%, and elevated at 2% or above. These thresholds are display rules, not validated risk categories.
