import pandas as pd

from src.backtesting.engine import performance_metrics, run_backtest


def test_signal_is_shifted_and_costed(prices):
    signal = pd.Series(0, index=prices.index, dtype=float)
    signal.iloc[10:] = 1
    result = run_backtest(prices, signal, transaction_cost_bps=10, slippage_bps=5)
    assert result.equity_curve.position.iloc[10] == 0
    assert result.equity_curve.position.iloc[11] == 1
    assert result.metrics["trades"] >= 1


def test_drawdown_metric():
    m = performance_metrics(pd.Series([0.1, -0.2, 0.05]))
    assert m["max_drawdown"] < 0


def test_metrics_are_json_safe_without_losses():
    metrics = performance_metrics(pd.Series([0.01, 0.02, 0.0]))
    assert metrics["profit_factor"] is None
    assert metrics["win_rate"] == 1.0
