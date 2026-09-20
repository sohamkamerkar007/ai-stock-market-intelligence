import numpy as np

from src.features import FEATURE_COLUMNS, build_features, build_target


def test_feature_pipeline_and_target(prices):
    features = build_features(prices)
    assert set(FEATURE_COLUMNS) <= set(features)
    assert features["rsi_14"].dropna().between(0, 100).all()
    targeted = build_target(features)
    expected = prices.close.shift(-1) / prices.close - 1
    np.testing.assert_allclose(targeted.future_return.iloc[:-1], expected.iloc[:-1])
    assert targeted.target_up.iloc[-1] is None or targeted.target_up.isna().iloc[-1]


def test_no_future_price_leakage(prices):
    before = build_features(prices)
    mutated = prices.copy()
    mutated.loc[350:, "close"] *= 10
    after = build_features(mutated)
    cols = FEATURE_COLUMNS
    assert before.loc[:349, cols].equals(after.loc[:349, cols])
