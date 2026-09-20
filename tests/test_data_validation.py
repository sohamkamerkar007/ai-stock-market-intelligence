import pytest

from src.data.providers import MarketDataError, validate_bars


def test_validation_deduplicates_and_rejects_bad_rows(prices):
    frame = prices.iloc[:4].copy()
    frame.loc[2, "high"] = 0
    frame = frame._append(frame.iloc[0], ignore_index=True)
    clean = validate_bars(frame)
    assert len(clean) == 3
    assert clean.timestamp.is_monotonic_increasing


def test_missing_column(prices):
    with pytest.raises(MarketDataError):
        validate_bars(prices.drop(columns="close"))
