# Research design

Primary question: does market context improve next-day directional classification beyond technical features for liquid Indian equities?

Hypotheses are tested through ablation, not assumed: A) technical; B) technical + discovered regime; C) technical + regime + time-aligned news sentiment; D) optional anomaly context. Each comparison holds the learner, split dates and metric definitions constant. Report statistical uncertainty across walk-forward folds and multiple market subperiods before drawing conclusions.

Secondary questions cover regime interpretability/persistence, anomaly concordance with known events, and risk-adjusted backtest changes. Because adjacent daily observations overlap through rolling features, naive IID significance tests are unsuitable. Prefer block bootstrap confidence intervals and Diebold–Mariano-style forecast comparisons where assumptions are defensible.

Experiment records are append-only `model_runs` with unique IDs. The Research Lab reads those records; it contains no showcase metrics.

