import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def prices():
    """Deterministic test fixture only; never used by the application or UI."""
    n = 420
    rng = np.random.default_rng(42)
    returns = np.r_[
        rng.normal(0.0005, 0.008, 140),
        rng.normal(-0.0004, 0.018, 140),
        rng.normal(0.0002, 0.006, 140),
    ]
    close = 100 * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.008, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.008, n))
    volume = rng.lognormal(14, 0.25, n)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
