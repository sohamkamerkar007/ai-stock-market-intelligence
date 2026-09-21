# Backtesting

Signals computed at a session close execute at the next session's open. Overnight returns belong to the previously held position; intraday returns belong to the new position. Costs (10 bps plus 5 bps slippage) are applied to turnover at entry/exit. The first 504 sessions are a common warm-up for every comparison, and all requested dates are filtered after retaining earlier feature and training history. Buy & Hold always uses the identical evaluation sessions.

Reported metrics: cumulative return, CAGR, Sharpe, Sortino, maximum drawdown (including the initial capital peak), annualised volatility, completed-trade win rate, entries, completed trades, exposure changes and completed-trade profit factor. Sortino uses the root mean square of negative returns across all sessions and a zero target return. Ratios without a denominator and win rates without completed trades return null, not fabricated zeros. Open positions remain marked to market; they are excluded from completed-trade statistics. Current policies are long/cash; ratios use 252 sessions and zero risk-free return. No taxes, market impact, liquidity/capacity constraints or point-in-time corporate-action corrections are included.

The API exposes buy-and-hold, SMA, walk-forward supervised ML, causal regime-aware ML and a sentiment-hybrid strategy. Regime states are fitted on preceding observations only. The hybrid returns an explicit unavailable response until time-aligned news exists for the asset. Research comparisons must use the identical date range, cost assumptions and asset universe.

Do not optimize only for return. Compare drawdown, volatility, turnover, stability by subperiod and sensitivity to costs. Report negative and null findings.
