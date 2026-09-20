from src.features import build_features
from src.regimes.discovery import causal_regime_labels, discover_regimes


def test_regime_discovery(prices):
    result = discover_regimes(build_features(prices), "gmm", 3)
    assert result.labels.nunique() == 3
    assert len(result.descriptions) == 3
    assert -1 <= result.metrics["silhouette"] <= 1
    assert -1 <= result.metrics["stability_ari"] <= 1


def test_causal_regimes_do_not_change_when_future_changes(prices):
    features = build_features(prices)
    before = causal_regime_labels(features, min_train=120, retrain_every=30)
    altered = prices.copy()
    altered.loc[330:, "close"] *= 1.8
    altered.loc[330:, "high"] = altered.loc[330:, ["open", "close"]].max(axis=1) * 1.01
    altered.loc[330:, "low"] = altered.loc[330:, ["open", "close"]].min(axis=1) * 0.99
    after = causal_regime_labels(build_features(altered), min_train=120, retrain_every=30)
    assert before.loc[:329].equals(after.loc[:329])
