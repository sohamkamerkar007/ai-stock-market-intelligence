# Four-page prediction methodology

## Latest available market observations

Yahoo Finance daily OHLCV observations are validated and stored incrementally. `POST /api/v1/market/refresh` fetches the latest available provider bars on demand. A daily trading date is not an intraday quote timestamp. Provider failure leaves the last verified stored observation visible with a freshness warning.

## Real-stock direction

`scripts.train_real_direction` fits Logistic Regression and XGBoost to actual tracked-stock OHLCV history for 1, 3, and 5 trading-session UP/DOWN targets. UP means future close-to-close return above zero; DOWN means zero or below. Technical inputs come from the single backward-looking `src.features.pipeline.build_features` definition. Same-session NIFTY 50 and same-sector returns, plus relative stock performance, provide market context known at the current close. Training and inference use the same persisted feature order, imputer, scaler, and classifier. No synthetic observations enter real-stock inference.

Dates are split 65% training, 15% validation, and 20% final test. Rows whose future target crosses a train or validation boundary are purged. The model is chosen by validation balanced accuracy, with F1 as a tie-breaker, then refit on eligible training and validation rows. The default horizon is selected from those same validation scores. The final test set is evaluated once. Selection does not use test metrics. The saved model is loaded on demand; requests never retrain it. SHAP attributions for the selected model are translated to plain-language reasons. The model's probability is displayed as model confidence, not as a calibrated success rate.

[real_direction_results.json](real_direction_results.json) contains every measured classification metric, split date, selected model, and baseline. Current real-stock test performance is near chance and without a demonstrated predictive advantage. The UI calls the direction result experimental and states that it has no demonstrated predictive advantage.

## Real-stock price and volatility

`scripts.train_final_stock_models` fits Lasso and Ridge per tracked stock and per horizon using the backward-looking `src.features.pipeline.expanded_features` definition. Targets are future close-to-close return and root mean square future daily log return. Future price is the latest actual close multiplied by one plus the estimated return.

The 65/15/20 chronological split is purged at both boundaries. Hyperparameters and the Lasso/Ridge choice are selected by validation MAE separately for price and volatility. The test set is untouched during selection. Saved artifacts hold the feature schema, preprocessing, model, validation errors, and held-out test metrics. [final_stock_regression_results.json](final_stock_regression_results.json) contains per-stock, per-horizon results. A high price-level R² does not establish direction forecasting skill.

## Controlled synthetic research

The older controlled synthetic classifier is a separate experiment documented in [SYNTHETIC_RESEARCH_DATASET.md](SYNTHETIC_RESEARCH_DATASET.md). Its approximately 81% test accuracy is not evidence of future Indian-stock accuracy. A training-majority constant baseline scores 43.68% on the test period; the 56.32% test-majority proportion uses hindsight and is not a deployable baseline. The requested 80% baseline target is unmet. Synthetic features are never fed into the real-stock inference endpoint.

## Similar-stock grouping

For each selected stock, the system first limits candidates to its authoritative sector. It calculates six real-history behavioral features and standardizes them within that sector. Sectors with at least four eligible stocks use K-Means; smaller sectors use a single nearest-peer group. Returned peers are always from the selected sector. Silhouette scores are included only when a meaningful partition exists. Group descriptions use actual group means.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m scripts.bootstrap --years 8
.\.venv\Scripts\python.exe -m scripts.train_real_direction
.\.venv\Scripts\python.exe -m scripts.train_final_stock_models
.\.venv\Scripts\python.exe -m pytest
```
