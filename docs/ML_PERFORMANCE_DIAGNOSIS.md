# Prediction performance diagnosis and experiments

## Findings before experimentation

The previous NIFTY display reported XGBoost accuracy 48.1%, F1 0.575 and ROC-AUC 0.481. Logistic Regression and Random Forest were also near chance. Around 2,000 observations per asset cover August 2018 to September 2026. Next-session UP prevalence is about 53.6% for NIFTY and close to 50% for the sampled equities. Imbalance is not the main cause. There are 45,901 price observations across 23 assets and no stored news articles in this snapshot.

The old validation partition was combined into training without any model selection. Settings were fixed; probability calibration, balanced accuracy, MCC, fold stability and market context were absent. Redundant inputs include return/log-return pairs and related moving averages. Missing volume is material for some index sessions. The old backtest logistic model was unscaled. An RSI edge-case returned 50 rather than 100 for uninterrupted gains. The UI incorrectly treated annualised volatility as daily volatility. These implementation issues are now corrected.

The underlying target remains noisy: today's OHLCV cannot encode overnight news or other unobserved next-day information. A desired 70% is a research target, not an empirically established achievable accuracy.

## Protocol declared before opening the new holdout

- Common warm-up and date coverage across feature sets; 37 dimensionless technical inputs (53 with market context), with three-index market returns/trend/volatility and relative strength where enabled.
- Four feature sets: technical, technical + market, technical + causal environment, technical + market + causal environment. No fake sentiment rows: the news experiment is skipped.
- Seven configurations across six model families: Logistic Regression (regularised and balanced), Random Forest, two regularised XGBoost configurations, Histogram Gradient Boosting and ExtraTrees.
- Horizons 1, 3 and 5 trading observations. Primary task remains one-day direction. The 0.15% threshold sensitivity records class balance, not a selected replacement label.
- Three expanding TimeSeriesSplit development folds; gap equals horizon. Assert that every training label becomes known before validation begins. Fit imputation and scaling inside each fold.
- 84 base configurations plus one development-only sigmoid-calibration comparison per horizon = 87 measured configurations (261 development fits, plus calibration internals and final evaluations).
- Freeze the one-day selection by development balanced accuracy, then Brier score. No threshold optimisation. No settings are changed based on final holdout results.
- Evaluate the final 15% only after selecting the configuration. Report every configuration, including poor outcomes. Fit the same selected configuration separately for all 23 assets.
- The earlier legacy holdout was already inspected. This is explicitly a retrospective holdout, not an untouched prospective test. Multiple comparisons and repeated use of historical data limit inference. Future observations are needed for prospective confirmation.

References: [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), [probability calibration](https://scikit-learn.org/stable/modules/calibration.html), [MCC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html).

## Actual outcomes

Selected: ExtraTrees, technical + market inputs, one-day horizon, no calibration. Development balanced accuracy 54.05%; retrospective NIFTY holdout accuracy 50.39%, balanced accuracy 50.28%, precision 50.61%, recall 63.85%, F1 0.5646, ROC-AUC 0.5326, MCC 0.0059 and Brier 0.2509 over 258 observations. The majority-class benchmark is also 50.39%: this is not evidence of useful classification skill.

The frozen legacy XGBoost on the identical 258 sessions scores accuracy 48.84%, F1 0.5742, ROC-AUC 0.4856. New accuracy and discrimination increase, but F1 decreases. It would be incorrect to claim every metric improved. Compared with the original 48.1% display, the evaluation start differs because the common context warm-up removes earlier sessions; the comparable legacy table addresses this difference. Legacy training also ended earlier than the new development boundary.

Best development balanced accuracy for three days is 50.08%; for five days, 50.83%. Neither justifies relabeling the primary task. One-day sigmoid calibration worsened development balanced accuracy to 47.87% and Brier to 0.2554; it is not deployed. Across assets, selected-configuration holdout accuracy ranges from 45.35% to 58.14%. All assets are displayed, not only winners.

Separate Ridge next-day return regression now provides real persisted estimates. NIFTY holdout MAE 0.00603, RMSE 0.00817, R² -0.0082, versus zero-return MAE 0.00595. This regression is experimental and not superior to its naive comparator. Its limitations appear with the UI prediction.

## Reproduction and provenance

Run `python -m scripts.run_research_matrix --publish`, followed by `python -m scripts.generate_predictions` and `python -m scripts.audit_research_results`. The first command performs model selection and publishes; the second uses saved selected models without retraining. `--legacy-train` explicitly requests the old pipeline. The audit compares legacy artifacts if present and calculates corrected historical strategies.

`docs/experiment_results.json` contains every fold, configuration, holdout, per-asset score, regime group, probability bin, feature diagnostic and backtest metric. `models/research_predictions.csv` contains actual held-out predictions; model bundles contain configuration, features and label cutoffs. Per-asset hashes detect input snapshot changes. Model files/raw data are not committed. Keep the locally permitted data snapshot to reproduce exact numbers; future provider corrections change results. The audit's frozen legacy comparison requires the original artifacts and is skipped on a new machine.

## Limits and next research steps

No 70% result was obtained. No causal evidence or market-beating conclusion is warranted. Cross-asset test outcomes are correlated and not independent samples. The displayed moving-block interval uses 10-session blocks and is descriptive. Current sector membership would introduce survivorship and membership bias; sector-specific features were not added without historical membership validation. Historical news coverage is absent. A chronological calibration comparison was made, but production probabilities remain uncalibrated. Market labels used for research features are expanding causal GMM states; the separate descriptive HMM timeline is retrospective.

Future work should prioritise point-in-time news, corporate actions, historical universe membership and a prospectively frozen evaluation over repeatedly retuning this same test period. Current OHLCV is provider-adjusted and may be revised; strict point-in-time corporate-action snapshots are not available.
