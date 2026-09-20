from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    metrics: dict[str, float]
    equity_curve: pd.DataFrame


def performance_metrics(returns: pd.Series, positions: pd.Series | None = None) -> dict[str, float]:
    returns = returns.dropna()
    equity = (1 + returns).cumprod()
    years = max(len(returns) / 252, 1 / 252)
    downside = returns[returns < 0].std(ddof=0)
    drawdown = equity / equity.cummax() - 1
    trades = int(positions.diff().abs().fillna(0).sum()) if positions is not None else 0
    wins, losses = returns[returns > 0].sum(), abs(returns[returns < 0].sum())
    return {
        "cumulative_return": float(equity.iloc[-1] - 1) if len(equity) else 0.0,
        "cagr": float(equity.iloc[-1] ** (1 / years) - 1) if len(equity) else 0.0,
        "sharpe": float(np.sqrt(252) * returns.mean() / returns.std(ddof=0))
        if returns.std(ddof=0)
        else 0.0,
        "sortino": float(np.sqrt(252) * returns.mean() / downside) if downside else 0.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "volatility": float(returns.std(ddof=0) * np.sqrt(252)),
        "win_rate": float((returns > 0).mean()),
        "trades": trades,
        "profit_factor": float(wins / losses) if losses else float("inf"),
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
