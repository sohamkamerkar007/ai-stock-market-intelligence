# Research design

Primary question: does market context improve next-day directional classification beyond technical features for liquid Indian equities?

Implemented ablations are A) technical; B) technical + causally assigned regime; and C) technical + causal regime + time-aligned news sentiment when real news exists. An anomaly-feature ablation was specified but is not implemented and is not claimed as a completed experiment. Each implemented comparison holds the learner, split dates and metric definitions constant.

Secondary questions cover regime interpretability/persistence, anomaly concordance with known events, and risk-adjusted backtest changes. Because adjacent daily observations overlap through rolling features, naive IID significance tests are unsuitable. Prefer block bootstrap confidence intervals and Diebold–Mariano-style forecast comparisons where assumptions are defensible.

Legacy real-market experiment records are append-only `model_runs` with unique IDs and remain accessible through API routes; there is no standalone Research Lab page. The four-page UI shows actual real-stock direction and Lasso/Ridge test metrics. The controlled synthetic benchmark is a separate research report and is not shown as a real-market prediction. See [real direction measurements](real_direction_results.json), [controlled synthetic measurements](final_classification_results.json), and [real-stock regression measurements](final_stock_regression_results.json).
