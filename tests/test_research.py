import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.backtesting.engine import performance_metrics, run_backtest
from src.features.pipeline import build_target, expanded_features
from src.ml.research import metrics


def test_expanded_features_future_invariance(prices):
    original, columns = expanded_features(prices)
    modified = prices.copy()
    modified.loc[350:, ["close", "open", "high", "low", "volume"]] *= 3
    changed, _ = expanded_features(modified)
    pd.testing.assert_frame_equal(original.loc[:349, columns], changed.loc[:349, columns])
    assert not {"target_up", "future_return", "label_available_at"}.intersection(columns)


def test_multiday_labels_are_purged(prices):
    for horizon in [1, 3, 5]:
        data = build_target(prices, horizon).dropna(subset=["target_up"])
        for train, test in TimeSeriesSplit(3, gap=horizon).split(data):
            assert data.iloc[train].label_available_at.max() < data.iloc[test].timestamp.min()
        assert data.future_return.iloc[0] == prices.close.iloc[horizon] / prices.close.iloc[0] - 1


def test_rsi_monotonic_increase(prices):
    prices["close"] = np.arange(len(prices)) + 100.0
    data, _ = expanded_features(prices)
    assert data.rsi_14.iloc[-1] == 100


def test_initial_loss_counts_in_drawdown_and_sortino():
    result = performance_metrics(pd.Series([-0.2, 0.1, 0]))
    assert result["max_drawdown"] == -0.2 or abs(result["max_drawdown"] + 0.2) < 1e-12
    assert np.isfinite(result["sortino"])


def test_entry_cannot_capture_overnight_gap():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3),
            "open": [100.0, 200.0, 200.0],
            "close": [100.0, 200.0, 200.0],
        }
    )
    result = run_backtest(frame, pd.Series([1.0, 1.0, 1.0]), 0, 0)
    assert result.metrics["cumulative_return"] == 0
    assert result.metrics["trades"] == 1
    assert result.metrics["closed_trades"] == 0
    assert result.metrics["win_rate"] is None


def test_majority_baseline_is_not_predictive_skill():
    result = metrics([0, 1, 1, 1, 1], [0.8] * 5)
    assert result["accuracy"] == 0.8
    assert result["balanced_accuracy"] == 0.5
    assert result["mcc"] == 0
