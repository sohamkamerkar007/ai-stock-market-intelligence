"""Saved real-market direction classifiers with chronological, purged evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features.pipeline import FEATURE_COLUMNS, build_features, build_target

DIRECTION_CONTEXT = ["index_return_1d", "sector_return_1d", "relative_to_market_1d", "relative_to_sector_1d"]


def _daily_returns(prices: pd.DataFrame) -> pd.Series:
    frame = prices.sort_values("timestamp").drop_duplicates("timestamp")
    return pd.Series(frame.close.pct_change(fill_method=None).to_numpy(), index=pd.to_datetime(frame.timestamp).dt.normalize())


def _add_context(frame: pd.DataFrame, index_returns: pd.Series, sector_returns: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(frame.timestamp).dt.normalize()
    frame["index_return_1d"] = dates.map(index_returns).to_numpy()
    frame["sector_return_1d"] = dates.map(sector_returns).to_numpy()
    frame["relative_to_market_1d"] = frame.return_1d - frame.index_return_1d
    frame["relative_to_sector_1d"] = frame.return_1d - frame.sector_return_1d
    return frame


def prepare_panel(observations: dict[str, pd.DataFrame], horizon: int,
                  sectors: dict[str, str] | None = None,
                  index_prices: pd.DataFrame | None = None) -> pd.DataFrame:
    if horizon not in (1, 3, 5):
        raise ValueError("Horizon must be 1, 3, or 5 trading sessions")
    frames = []
    index_returns = _daily_returns(index_prices) if index_prices is not None else None
    sector_returns = {}
    if sectors and index_returns is not None:
        for sector in set(sectors.values()):
            series = [_daily_returns(prices).rename(symbol) for symbol, prices in observations.items()
                      if sectors.get(symbol) == sector]
            if series:
                sector_returns[sector] = pd.concat(series, axis=1).mean(axis=1)
    for symbol, prices in observations.items():
        if len(prices) < 300:
            continue
        frame = build_target(build_features(prices), horizon)
        if index_returns is not None and sectors and sector_returns.get(sectors.get(symbol)) is not None:
            frame = _add_context(frame, index_returns, sector_returns[sectors[symbol]])
        frame["symbol"] = symbol
        frames.append(frame)
    if not frames:
        raise ValueError("Insufficient real-market history")
    return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"])


def split_panel(panel: pd.DataFrame, features: list[str] | None = None):
    dates = sorted(panel.timestamp.unique())
    validation_start, test_start = dates[int(len(dates) * .65)], dates[int(len(dates) * .8)]
    eligible = panel.dropna(subset=["target_up", *(features or FEATURE_COLUMNS)])
    train = eligible[(eligible.timestamp < validation_start) & (eligible.label_available_at < validation_start)]
    validation = eligible[(eligible.timestamp >= validation_start) & (eligible.timestamp < test_start) & (eligible.label_available_at < test_start)]
    test = eligible[eligible.timestamp >= test_start]
    if min(len(train), len(validation), len(test)) < 100:
        raise ValueError("Insufficient chronological observations")
    return train, validation, test


def _candidate(name: str):
    classifier = (LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
                  if name == "logistic_regression" else
                  XGBClassifier(n_estimators=160, max_depth=3, learning_rate=.05,
                                subsample=.8, colsample_bytree=.8, eval_metric="logloss",
                                random_state=42, n_jobs=2))
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), classifier)


def _metrics(model, frame, features):
    truth = frame.target_up.astype(int)
    probability = model.predict_proba(frame[features])[:, 1]
    prediction = (probability >= .5).astype(int)
    return {"accuracy": float(accuracy_score(truth, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "precision": float(precision_score(truth, prediction, zero_division=0)),
            "recall": float(recall_score(truth, prediction, zero_division=0)),
            "f1": float(f1_score(truth, prediction, zero_division=0)),
            "roc_auc": float(roc_auc_score(truth, probability)) if truth.nunique() > 1 else None,
            "confusion_matrix": confusion_matrix(truth, prediction, labels=[0, 1]).tolist(),
            "rows": len(frame)}


def train(observations: dict[str, pd.DataFrame], horizon: int,
          sectors: dict[str, str] | None = None,
          index_prices: pd.DataFrame | None = None) -> dict:
    features = list(FEATURE_COLUMNS) + (DIRECTION_CONTEXT if sectors and index_prices is not None else [])
    panel = prepare_panel(observations, horizon, sectors, index_prices)
    training, validation, test = split_panel(panel, features)
    results = {}
    for name in ("logistic_regression", "xgboost"):
        candidate = _candidate(name).fit(training[features], training.target_up.astype(int))
        validation_metrics = _metrics(candidate, validation, features)
        final = _candidate(name).fit(
            pd.concat([training, validation])[features],
            pd.concat([training, validation]).target_up.astype(int))
        results[name] = {"validation": validation_metrics, "test": _metrics(final, test, features),
                         "pipeline": final}
    selected = max(results, key=lambda name: (results[name]["validation"]["balanced_accuracy"],
                                              results[name]["validation"]["f1"]))
    return {"version": 1, "data_type": "real_indian_market", "horizon": horizon,
            "features": features, "selected": selected, "models": results,
            "train_end": str(training.timestamp.max().date()),
            "validation_end": str(validation.timestamp.max().date()),
            "test_start": str(test.timestamp.min().date()),
            "test_end": str(test.timestamp.max().date()),
            "baseline_test_accuracy": float((test.target_up.astype(int) ==
                                              int(training.target_up.mean() >= .5)).mean())}


FEATURE_LABELS = {
    "return_1d": "Latest price move", "log_return_1d": "Latest price move",
    "return_5d": "Recent price performance",
    "return_3d": "Recent price momentum", "return_10d": "Two-week price momentum",
    "return_20d": "Monthly price trend", "rsi_14": "Recent buying momentum",
    "macd": "Short-term trend signal", "momentum_10": "Recent price momentum",
    "volatility_20": "Recent price instability", "volume_change": "Trading activity",
    "relative_volume_20": "Trading activity versus usual", "sma_ratio_20": "Price versus recent average",
    "sma_ratio_10": "Price versus short-term average",
    "sma_ratio_5": "Price versus short-term average", "sma_ratio_50": "Price versus longer-term average",
    "ema_ratio_12": "Price versus recent trend",
    "ma_cross_10_20": "Short-term versus longer trend",
    "atr_14": "Typical daily price movement",
    "bb_width_20": "Recent trading range",
    "drawdown_60": "Distance below recent high",
    "volatility_change": "Change in recent price instability",
    "volume_momentum_5": "Change in trading activity",
    "index_return_1d": "Overall market movement", "sector_return_1d": "Movement in the stock's sector",
    "relative_to_market_1d": "Performance versus the overall market",
    "relative_to_sector_1d": "Performance versus sector peers",
}


def infer(artifact: dict, prices: pd.DataFrame, index_prices: pd.DataFrame | None = None,
          sector_observations: dict[str, pd.DataFrame] | None = None) -> dict:
    frame = build_features(prices)
    if DIRECTION_CONTEXT[0] in artifact["features"]:
        if index_prices is None or not sector_observations:
            raise ValueError("Current market and sector history is required for direction analysis")
        sector_returns = pd.concat([_daily_returns(p).rename(s) for s,p in sector_observations.items()],axis=1).mean(axis=1)
        frame = _add_context(frame, _daily_returns(index_prices), sector_returns)
    latest = frame.iloc[-1]
    if pd.isna(latest[artifact["features"]]).any():
        raise ValueError("Insufficient recent real-market observations for direction analysis")
    values = latest[artifact["features"]].to_frame().T.astype(float)
    name = artifact["selected"]
    pipeline = artifact["models"][name]["pipeline"]
    probability_up = float(pipeline.predict_proba(values)[0, 1])
    transformed = pipeline[:-1].transform(values)
    classifier = pipeline[-1]
    try:
        if name == "xgboost":
            contributions = shap.TreeExplainer(classifier).shap_values(transformed)[0]
        else:
            contributions = shap.LinearExplainer(
                classifier, np.zeros((1, transformed.shape[1]))).shap_values(transformed)[0]
        contributions = np.asarray(contributions).reshape(-1)
    except Exception:
        contributions = np.array([])
    direction = "UP" if probability_up >= .5 else "DOWN"
    signed = contributions if direction == "UP" else -contributions
    strongest = sorted(range(len(signed)), key=lambda i: abs(signed[i]), reverse=True)[:5]
    reasons = [{"label": FEATURE_LABELS.get(artifact["features"][i],
                                      artifact["features"][i].replace("_", " ").title()),
                "feature": artifact["features"][i],
                "effect": "supports" if signed[i] > 0 else "reduces confidence",
                "value": float(signed[i]), "feature_value": float(values.iloc[0, i])} for i in strongest]
    return {"direction": direction, "probability_up": probability_up,
            "confidence": max(probability_up, 1 - probability_up), "model": name,
            "horizon": artifact["horizon"], "as_of": str(pd.Timestamp(latest.timestamp).date()),
            "reasons": reasons, "explanation_available": bool(reasons),
            "feature_count": len(artifact["features"]),
            "train_end": artifact["train_end"], "validation_end": artifact["validation_end"],
            "test_start": artifact["test_start"], "test_end": artifact["test_end"],
            "test_metrics": artifact["models"][name]["test"],
            "baseline_test_accuracy": artifact["baseline_test_accuracy"]}
