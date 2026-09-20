# Backtesting

Signals computed from close-of-day features are shifted one period before being applied to returns. This is the central look-ahead safeguard. Costs are charged on absolute position changes using configurable transaction-cost and slippage basis points.

Reported metrics: cumulative return, annualized CAGR, Sharpe, Sortino, maximum drawdown, annualized volatility, win rate, number of position changes and profit factor. Metrics are descriptive and do not include taxes, market impact, liquidity constraints, corporate-action errors or capacity.

The API exposes buy-and-hold, SMA, walk-forward supervised ML, regime-aware ML and a sentiment-hybrid strategy. The hybrid returns an explicit unavailable response until time-aligned news exists for the asset. Research comparisons must use the identical date range, cost assumptions and asset universe.

Do not optimize only for return. Compare drawdown, volatility, turnover, stability by subperiod and sensitivity to costs. Report negative and null findings.
