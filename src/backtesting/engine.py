from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    metrics: dict[str, Any]
    equity_curve: pd.DataFrame


def performance_metrics(returns: pd.Series, positions: pd.Series | None = None) -> dict[str, Any]:
    returns = returns.dropna()
    if returns.empty:
        raise ValueError("Cannot calculate performance metrics for an empty return series")
    equity = (1 + returns).cumprod()
    years = max(len(returns) / 252, 1 / 252)
    downside = returns[returns < 0].std(ddof=0)
    drawdown = equity / equity.cummax() - 1
    trades = (
        int((positions.diff().abs().fillna(positions.abs()) > 1e-12).sum())
        if positions is not None
        else 0
    )
    active_returns = returns[returns != 0]
    wins, losses = (
        active_returns[active_returns > 0].sum(),
        abs(active_returns[active_returns < 0].sum()),
    )
    final_equity = float(equity.iloc[-1])
    return {
        "cumulative_return": final_equity - 1,
        "cagr": float(final_equity ** (1 / years) - 1) if final_equity > 0 else -1.0,
        "sharpe": float(np.sqrt(252) * returns.mean() / returns.std(ddof=0))
        if returns.std(ddof=0)
        else 0.0,
        "sortino": float(np.sqrt(252) * returns.mean() / downside) if downside else 0.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "volatility": float(returns.std(ddof=0) * np.sqrt(252)),
        "win_rate": float((active_returns > 0).mean()) if len(active_returns) else 0.0,
        "trades": trades,
        "profit_factor": float(wins / losses) if losses else None,
    }


def run_backtest(
    frame: pd.DataFrame,
    signal: pd.Series,
    transaction_cost_bps: float = 10,
    slippage_bps: float = 5,
) -> BacktestResult:
    """Signals formed at close t are shifted and applied to return t+1."""
    data = frame.sort_values("timestamp").copy()
    position = signal.reindex(data.index).fillna(0).clip(-1, 1).shift(1).fillna(0)
    turnover = position.diff().abs().fillna(position.abs())
    costs = turnover * (transaction_cost_bps + slippage_bps) / 10_000
    returns = position * data["close"].pct_change().fillna(0) - costs
    equity = pd.DataFrame(
        {
            "timestamp": data["timestamp"],
            "strategy_return": returns,
            "equity": (1 + returns).cumprod(),
            "position": position,
        }
    )
    return BacktestResult(performance_metrics(returns, position), equity)
