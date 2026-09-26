import numpy as np
import pandas as pd

from src.ml.synthetic_research import (
    FEATURE_GROUPS,
    SyntheticSpec,
    build_synthetic_features,
    chronological_panel_split,
    generate_synthetic_market,
)


def test_synthetic_generator_is_reproducible_and_multi_asset():
    spec = SyntheticSpec(seed=7, sessions=120, assets_per_sector=1)
    first = generate_synthetic_market(spec)
    second = generate_synthetic_market(spec)
    pd.testing.assert_frame_equal(first, second)
    assert first.symbol.nunique() == 8
    assert first.sector.nunique() == 8
    assert not first.duplicated(["timestamp", "symbol"]).any()


def test_synthetic_features_do_not_depend_on_later_rows():
    raw = generate_synthetic_market(SyntheticSpec(seed=11, sessions=180, assets_per_sector=1))
    original = build_synthetic_features(raw)
    cutoff = raw.timestamp.drop_duplicates().sort_values().iloc[120]
    changed_raw = raw.copy()
    changed_raw.loc[changed_raw.timestamp > cutoff, ["open", "high", "low", "close", "adjusted_close", "volume"]] *= 3
    changed = build_synthetic_features(changed_raw)
    columns = FEATURE_GROUPS["technical_market_regime_volume"]
    before = original[original.timestamp <= cutoff].sort_values(["timestamp", "symbol"])[columns]
    after = changed[changed.timestamp <= cutoff].sort_values(["timestamp", "symbol"])[columns]
    pd.testing.assert_frame_equal(before, after)


def test_synthetic_targets_are_forward_and_split_is_purged():
    raw = generate_synthetic_market(SyntheticSpec(seed=12, sessions=180, assets_per_sector=1))
    frame = build_synthetic_features(raw)
    sample = frame.dropna(subset=["target_3d"]).iloc[100]
    asset = frame[frame.symbol == sample.symbol].sort_values("timestamp").reset_index(drop=True)
    index = int(asset.index[asset.timestamp == sample.timestamp][0])
    expected = asset.close.iloc[index + 3] / asset.close.iloc[index] - 1
    assert np.isclose(sample.future_return_3d, expected)
    train, validation, test = chronological_panel_split(frame, 3)
    assert train.target_available_at_3d.max() < validation.timestamp.min()
    assert validation.target_available_at_3d.max() < test.timestamp.min()
    assert train.timestamp.max() < validation.timestamp.min() < test.timestamp.min()
    assert not any("target" in column or "future" in column for column in FEATURE_GROUPS["technical_market_regime_volume"])
