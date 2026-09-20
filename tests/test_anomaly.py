from src.anomaly.detection import detect_anomalies
from src.features import build_features


def test_anomaly_detection(prices):
    prices.loc[300, "close"] *= 1.3
    prices.loc[300, "high"] = max(prices.loc[300, ["open", "close"]]) * 1.01
    result = detect_anomalies(build_features(prices))
    assert {"anomaly_score", "anomaly_type", "severity"} <= set(result)
    assert result.is_anomaly.any()
    assert result.loc[300, "anomaly_type"] == "abnormal_return"
