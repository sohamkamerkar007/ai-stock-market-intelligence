"""Time-aware final-project prediction pipelines and persisted inference artifacts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# Dimensionless inputs avoid fitting spurious absolute price/volume levels that
# drift strongly between chronological periods and differ across assets.
FINAL_FEATURES = [
    "return_1d", "return_3d", "return_5d", "return_10d", "log_return",
    "price_to_sma20", "price_to_sma50", "trend_strength", "rsi_14",
    "momentum_5d", "momentum_10d", "roc_10", "rolling_volatility_20",
    "atr_percentage", "bollinger_width", "volume_change", "relative_volume",
    "volume_zscore", "market_return", "market_momentum", "market_volatility",
    "sector_return", "sector_momentum", "market_environment_code",
    "regime_duration", "regime_strength", "volatility_regime_code",
]


@lru_cache(maxsize=1)
def load_research_panel() -> pd.DataFrame:
    path = ROOT / "data/processed/final_prediction_research.csv.gz"
    if not path.exists():
        raise FileNotFoundError("Generate the controlled dataset with python -m scripts.generate_final_research_dataset")
    return pd.read_csv(path, parse_dates=["timestamp", "target_available_at_1d", "target_available_at_3d", "target_available_at_5d"])


def development_split(frame: pd.DataFrame, horizon: int):
    """First 50% train, next 20% validation, last 30% sequestered test."""
    if horizon not in (1, 3, 5):
        raise ValueError("Unsupported prediction horizon")
    dates = sorted(frame.timestamp.unique())
    validation_start = dates[int(len(dates) * 0.50)]
    test_start = dates[int(len(dates) * 0.70)]
    target = f"target_{horizon}d"
    available = f"target_available_at_{horizon}d"
    eligible = frame.dropna(subset=[target])
    train = eligible[(eligible.timestamp < validation_start) & (eligible[available] < validation_start)].copy()
    validation = eligible[(eligible.timestamp >= validation_start) & (eligible.timestamp < test_start) & (eligible[available] < test_start)].copy()
    test = eligible[eligible.timestamp >= test_start].copy()
    assert train[available].max() < validation.timestamp.min()
    assert validation[available].max() < test.timestamp.min()
    return train, validation, test
