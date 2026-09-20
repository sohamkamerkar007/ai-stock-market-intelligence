# Limitations

- Free historical access is unofficial and has no guaranteed SLA; licensed data is needed for operational or redistributed use.
- The starting universe has survivorship bias and does not reconstruct historical index membership.
- Daily bars cannot model intraday execution, auction behavior, price limits, spreads or impact.
- Corporate actions and provider revisions can change historical observations.
- The market-hours check does not yet encode all Indian exchange holidays/special sessions.
- VADER is a transparent baseline, not a finance-native transformer. Optional FinBERT is heavier and must be validated on Indian financial language.
- Entity matching is dictionary-based and can miss aliases or create ambiguous associations.
- Regime states are sample- and parameter-dependent; human-readable labels summarize observations rather than objective economic truth.
- SHAP describes model attribution, not causality.
- Backtests omit taxes, borrow constraints, partial fills, liquidity/capacity and portfolio-level risk controls.
- A statistically positive holdout result does not establish future profitability. Multiple testing, nonstationarity and data snooping remain material risks.

Recommended improvements: point-in-time index membership, licensed corporate-action-adjusted data, exchange holiday calendars, purged/embargoed cross-validation, block-bootstrap uncertainty, calibrated probabilities, richer entity linking, a validated India-finance sentiment corpus, and paper-trading observation without order submission.
