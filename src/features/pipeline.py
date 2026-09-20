import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "return_1d",
    "log_return_1d",
    "return_5d",
    "return_20d",
    "sma_ratio_10",
    "sma_ratio_20",
    "ema_ratio_12",
    "ma_cross_10_20",
    "rsi_14",
    "macd",
    "momentum_10",
    "volatility_20",
    "atr_14",
    "bb_width_20",
    "volume_change",
    "relative_volume_20",
    "drawdown_60",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def build_features(
    prices: pd.DataFrame,
    index_returns: pd.Series | None = None,
    sector_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Build close-of-day features. Rolling windows are backward-looking only."""
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if missing := required - set(prices.columns):
        raise ValueError(f"Missing price columns: {sorted(missing)}")
    frame = prices.copy().sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    c, h, low, vol = (
        frame["close"].astype(float),
        frame["high"].astype(float),
        frame["low"].astype(float),
        frame["volume"].astype(float),
    )
    frame["return_1d"] = c.pct_change()
    frame["log_return_1d"] = np.log(c).diff()
    frame["return_5d"] = c.pct_change(5)
    frame["return_20d"] = c.pct_change(20)
    sma10, sma20 = c.rolling(10, min_periods=10).mean(), c.rolling(20, min_periods=20).mean()
    frame["sma_ratio_10"] = c / sma10 - 1
    frame["sma_ratio_20"] = c / sma20 - 1
    frame["ema_ratio_12"] = c / c.ewm(span=12, adjust=False).mean() - 1
    frame["ma_cross_10_20"] = sma10 / sma20 - 1
    frame["rsi_14"] = _rsi(c)
    frame["macd"] = (c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()) / c
    frame["momentum_10"] = c / c.shift(10) - 1
    frame["volatility_20"] = frame["log_return_1d"].rolling(20, min_periods=20).std() * np.sqrt(252)
    previous = c.shift(1)
    true_range = pd.concat([h - low, (h - previous).abs(), (low - previous).abs()], axis=1).max(
        axis=1
    )
    frame["atr_14"] = true_range.rolling(14, min_periods=14).mean() / c
    std20 = c.rolling(20, min_periods=20).std()
    frame["bb_width_20"] = 4 * std20 / sma20
    frame["volume_change"] = vol.pct_change().replace([np.inf, -np.inf], np.nan)
    frame["relative_volume_20"] = vol / vol.rolling(20, min_periods=20).mean().replace(0, np.nan)
    frame["drawdown_60"] = c / c.rolling(60, min_periods=20).max() - 1
    if index_returns is not None:
        frame["index_return_1d"] = index_returns.reindex(frame.index)
    if sector_returns is not None:
        frame["sector_return_1d"] = sector_returns.reindex(frame.index)
    return frame.replace([np.inf, -np.inf], np.nan)


def build_target(feature_frame: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Attach next-period target; the final unavailable target is removed, never imputed."""
    frame = feature_frame.copy()
    frame["future_return"] = frame["close"].shift(-horizon) / frame["close"] - 1
    frame["target_up"] = (frame["future_return"] > 0).astype("Int64")
    frame.loc[frame["future_return"].isna(), "target_up"] = pd.NA
    return frame
