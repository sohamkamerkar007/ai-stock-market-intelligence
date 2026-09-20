from pathlib import Path

from src.features import FEATURE_COLUMNS, build_features, build_target
from src.ml.training import chronological_split, train_classifier


def test_chronological_training(prices, tmp_path: Path):
    data = build_target(build_features(prices)).dropna(subset=FEATURE_COLUMNS)
    a, b, c = chronological_split(data)
    assert a.timestamp.max() < b.timestamp.min() < c.timestamp.min()
    result = train_classifier(data, FEATURE_COLUMNS, "logistic_regression", tmp_path)
    assert set(result.metrics) >= {"accuracy", "f1", "roc_auc"}
    assert Path(result.artifact_path).exists()
