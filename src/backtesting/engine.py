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
    downside = float(np.sqrt(np.mean(np.minimum(returns, 0) ** 2)))
    drawdown = equity / equity.cummax().clip(lower=1.0) - 1
    position_changes = (
        int((positions.diff().abs().fillna(positions.abs()) > 1e-12).sum())
        if positions is not None
        else 0
    )
    active_returns = returns[returns != 0]
    completed = []
    entries = 0
    if positions is not None:
        positions = positions.reindex(returns.index).fillna(0)
        previous, growth = 0.0, 1.0
        for exposure, result in zip(positions, returns, strict=True):
            if exposure != 0 and previous == 0:
                entries += 1
                growth = 1.0
            if exposure != 0 or previous != 0:
                growth *= 1 + result
            if previous != 0 and exposure == 0:
                completed.append(growth - 1)
                growth = 1.0
            previous = exposure
        # An open position is marked to market but is not a completed trade.
        trade_returns = pd.Series(completed, dtype=float)
    else:
        trade_returns = active_returns
    wins, losses = (
        trade_returns[trade_returns > 0].sum(),
        abs(trade_returns[trade_returns < 0].sum()),
    )
    final_equity = float(equity.iloc[-1])
    return {
        "cumulative_return": final_equity - 1,
        "cagr": float(final_equity ** (1 / years) - 1) if final_equity > 0 else -1.0,
        "sharpe": float(np.sqrt(252) * returns.mean() / returns.std(ddof=0))
        if returns.std(ddof=0)
        else 0.0,
        "sortino": float(np.sqrt(252) * returns.mean() / downside) if downside else None,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "volatility": float(returns.std(ddof=0) * np.sqrt(252)),
        "win_rate": float((trade_returns > 0).mean()) if len(trade_returns) else None,
        "profitable_day_rate": float((active_returns > 0).mean()) if len(active_returns) else None,
        "trades": entries,
        "closed_trades": len(completed),
        "position_changes": position_changes,
        "profit_factor": float(wins / losses) if losses else None,
    }


def run_backtest(
    frame: pd.DataFrame,
    signal: pd.Series,
    transaction_cost_bps: float = 10,
    slippage_bps: float = 5,
) -> BacktestResult:
    """Close-t decisions execute at the next open, then mark to that session's close.

    Overnight movement belongs to the previous exposure. This avoids capturing a
    gap which occurred before a new position could actually be entered.
    """
    data = frame.sort_values("timestamp").copy()
    position = signal.reindex(data.index).fillna(0).clip(-1, 1).shift(1).fillna(0)
    turnover = position.diff().abs().fillna(position.abs())
    costs = turnover * (transaction_cost_bps + slippage_bps) / 10_000
    if transaction_cost_bps < 0 or slippage_bps < 0:
        raise ValueError("Costs must be non-negative")
    overnight = (data["open"] / data["close"].shift(1) - 1).fillna(0)
    intraday = data["close"] / data["open"] - 1
    old_position = position.shift(1).fillna(0)
    returns = (1 + old_position * overnight) * (1 - costs) * (1 + position * intraday) - 1
    equity = pd.DataFrame(
        {
            "timestamp": data["timestamp"],
            "strategy_return": returns,
            "equity": (1 + returns).cumprod(),
            "position": position,
        }
    )
    return BacktestResult(performance_metrics(returns, position), equity)
