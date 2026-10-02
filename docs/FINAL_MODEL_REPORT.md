# Final model report (2 October 2026)

All figures below are computed from saved reports and artifacts. Real-market and controlled synthetic results are separate evaluations. No figure implies a guaranteed future trading outcome.

## Real-market direction classification

Training and inference use actual tracked-stock daily OHLCV and the same `build_features` schema. The split is chronological 65/15/20 with purging across boundaries. Highest validation F1, then validation accuracy, chooses the saved model. XGBoost is selected for all three horizons. Test data did not influence that choice.

| Horizon | Model | Test accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | Training-majority baseline accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 day | XGBoost | 49.49% | 50.12% | 48.22% | 66.88% | 56.03% | 50.14% | 48.13% |
| 3 days | XGBoost | 49.26% | 50.44% | 48.37% | 81.51% | 60.72% | 50.14% | 48.10% |
| 5 days | XGBoost | 48.75% | 49.94% | 48.44% | 89.13% | 62.77% | 51.12% | 48.48% |

These scores are near chance. Model probabilities and SHAP explanations are real computations, but the classifier has no demonstrated predictive advantage. Full validation and test metrics for **both** Logistic Regression and XGBoost are in [real_direction_results.json](real_direction_results.json).

## Controlled synthetic classification

The controlled 3-session panel has 15,768 chronological held-out rows. Its training-majority UP baseline, fixed before looking at test labels, scores 43.68% accuracy, 50.00% balanced accuracy, 43.68% precision, 100.00% recall, 60.81% F1, and 50.00% ROC-AUC. The hindsight test-majority proportion is 56.32%; using that class as a deployed baseline would use test-label knowledge.

| Model | Test accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | Validation F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 81.32% | 81.04% | 78.54% | 78.77% | 78.65% | 88.95% | 89.25% |
| XGBoost | 81.37% | 81.11% | 78.44% | 79.08% | 78.76% | 88.88% | 87.47% |

Logistic Regression wins the prespecified validation F1 comparison for the **synthetic research panel**. Neither synthetic classifier is used to infer the selected real stock. The requested 80% baseline threshold is not met. Full metrics are in [final_classification_results.json](final_classification_results.json).

## Real-stock price and volatility regression

For each stock and 1/3/5-session horizon, Lasso and Ridge are compared by validation MAE separately for future return and future daily RMS volatility. The selected model is persisted. As a concrete example, RELIANCE at the 1-session horizon selects Lasso for both targets. On its held-out period, Lasso price MAE is ₹13.29, RMSE ₹17.85, MAPE 0.97%, and R² 0.964; Ridge price MAE is ₹13.70, RMSE ₹18.37, MAPE 1.00%, and R² 0.962. For daily volatility, Lasso MAE is 0.00674, RMSE 0.00869, and R² 0.014; Ridge MAE is 0.00678, RMSE 0.00871, and R² 0.008. Volatility MAPE is undefined on that test window because actual zero values occur. The near-zero volatility R² limits the usefulness of those estimates. Every stock and horizon is in [final_stock_regression_results.json](final_stock_regression_results.json).

## Sector-constrained similarity

The method standardizes six backward-looking behavioral features within the selected sector. Banking has 3 K-Means groups and silhouette score 0.386. The remaining sectors have fewer than 4 tracked eligible stocks, so each uses a single nearest-peer group and has no meaningful silhouette score. Direct API verification: RELIANCE returns only Energy peers ONGC and NTPC; HDFCBANK returns only Banking peer AXISBANK; TCS returns only IT peers WIPRO and INFY.

## Market data and verification

An incremental provider refresh on 2 October 2026 received 230 real bars, wrote 207 new observations, and reported no failures. The latest stored trading session is 1 October 2026. These are daily observations, not live ticks. The complete suite passed 34 tests and Ruff passed at the final verification point; see the run output for the exact current count if tests are added later.
