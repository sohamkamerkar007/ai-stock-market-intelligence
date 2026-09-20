from src.features import build_features
from src.regimes.discovery import discover_regimes


def test_regime_discovery(prices):
    result = discover_regimes(build_features(prices), "gmm", 3)
    assert result.labels.nunique() == 3
    assert len(result.descriptions) == 3
    assert -1 <= result.metrics["silhouette"] <= 1
