"""Chronological real-price regression; never uses synthetic labels or artifacts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features.pipeline import expanded_features

FEATURES = [
    "return_1d", "return_5d", "return_20d", "sma_ratio_10", "sma_ratio_20",
    "ema_ratio_12", "ma_cross_10_20", "rsi_14", "macd", "momentum_10",
    "volatility_20", "atr_14", "bb_width_20", "relative_volume_20",
    "drawdown_60", "return_2d", "return_3d", "return_10d", "sma_ratio_50",
    "ema_ratio_50", "return_acceleration", "rsi_change_3", "macd_histogram",
    "distance_low_20", "trend_slope_20", "volatility_ratio", "volume_zscore",
    "price_volume_interaction", "intraday_return", "overnight_gap",
]


def prepare(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if horizon not in (1, 3, 5):
        raise ValueError("Horizon must be 1, 3, or 5 sessions")
    frame, _ = expanded_features(prices)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame["future_return"] = frame.close.shift(-horizon) / frame.close - 1
    logs = np.log(frame.close).diff()
    frame["future_volatility"] = np.sqrt(
        sum(logs.shift(-step).pow(2) for step in range(1, horizon + 1)) / horizon
    )
    frame["label_available_at"] = frame.timestamp.shift(-horizon)
    return frame.replace([np.inf, -np.inf], np.nan)


def split(frame: pd.DataFrame):
    """Chronological 65/15/20 date split; purge labels crossing each boundary."""
    eligible = frame.dropna(subset=["future_return", "future_volatility", "label_available_at"])
    dates = sorted(frame.timestamp.unique())
    val_start, test_start = dates[int(len(dates) * .65)], dates[int(len(dates) * .80)]
    train = eligible[(eligible.timestamp < val_start) & (eligible.label_available_at < val_start)]
    validation = eligible[(eligible.timestamp >= val_start) & (eligible.timestamp < test_start) & (eligible.label_available_at < test_start)]
    test = eligible[eligible.timestamp >= test_start]
    if min(len(train), len(validation), len(test)) < 30:
        raise ValueError("At least 300 chronological price observations are required")
    return train, validation, test


def _model(kind: str, alpha: float):
    estimator = Lasso(alpha=alpha, max_iter=50000, tol=1e-3) if kind == "lasso" else Ridge(alpha=alpha)
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), estimator)


def _metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape": float(np.mean(np.abs((actual - predicted) / actual))) if np.all(actual != 0) else None,
        "r2": float(r2_score(actual, predicted)),
        "n": int(len(actual)),
    }


def train(prices: pd.DataFrame, horizon: int) -> dict:
    frame = prepare(prices, horizon)
    training, validation, test = split(frame)
    trainval = pd.concat([training, validation])
    result = {"horizon": horizon, "features": FEATURES, "models": {},
              "test_start": str(pd.Timestamp(test.timestamp.min()).date()),
              "test_end": str(pd.Timestamp(test.timestamp.max()).date()),
              "latest_observation": str(pd.Timestamp(frame.timestamp.max()).date())}
    for target in ("future_return", "future_volatility"):
        for kind in ("lasso", "ridge"):
            candidates = (0.00001, 0.0001, 0.001) if kind == "lasso" else (0.1, 1.0, 10.0)
            best_alpha = min(candidates, key=lambda alpha: mean_absolute_error(
                validation[target], _model(kind, alpha).fit(training[FEATURES], training[target]).predict(validation[FEATURES])
            ))
            validation_model = _model(kind, best_alpha).fit(training[FEATURES], training[target])
            validation_mae = float(mean_absolute_error(
                validation[target], validation_model.predict(validation[FEATURES])))
            model = _model(kind, best_alpha).fit(trainval[FEATURES], trainval[target])
            predicted = model.predict(test[FEATURES])
            if target == "future_return":
                actual_value = test.close.to_numpy() * (1 + test[target].to_numpy())
                predicted_value = test.close.to_numpy() * (1 + predicted)
                metric_name = "future_price_rupees"
            else:
                actual_value, predicted_value = test[target], predicted
                metric_name = "future_daily_rms_log_return"
            result["models"][f"{kind}_{target}"] = {
                "model": model, "alpha": best_alpha, "validation_mae": validation_mae,
                "metrics": {metric_name: _metrics(actual_value, predicted_value)},
            }
    return result


def infer(artifact: dict, prices: pd.DataFrame) -> dict:
    frame = prepare(prices, artifact["horizon"])
    latest = frame.iloc[-1]
    input_frame = frame.tail(1)[FEATURES]
    close = float(latest.close)
    estimates = {}
    for name, entry in artifact["models"].items():
        value = float(entry["model"].predict(input_frame)[0])
        estimates[name] = max(0.0, close * (1 + value)) if "future_return" in name else max(0.0, value)
    selected = {}
    for target in ("future_return", "future_volatility"):
        candidates = {name: entry for name, entry in artifact["models"].items()
                      if name.endswith(target) and "validation_mae" in entry}
        if candidates:
            name = min(candidates, key=lambda key: candidates[key]["validation_mae"])
            selected[target] = {"model": name.split("_")[0], "estimate": estimates[name],
                                "validation_mae": candidates[name]["validation_mae"]}
    return {"as_of": str(pd.Timestamp(latest.timestamp).date()), "current_close": close,
            "horizon": artifact["horizon"], "estimates": estimates, "selected": selected,
            "test_start": artifact["test_start"], "test_end": artifact["test_end"],
            "metrics": {name: entry["metrics"] for name, entry in artifact["models"].items()}}
