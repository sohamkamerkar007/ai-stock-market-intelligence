"""Controlled synthetic market-like data and leakage-safe research experiments.

This module is intentionally isolated from the real-market data pipeline.  It creates
an observable, persistent market process and labels each row from the *next* simulated
session.  The generator is useful for demonstrating the ML methodology when noisy,
real next-day OHLCV data does not contain a reliable signal.  It is not a market
simulator, investment model, or source of real-market performance claims.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.ml.research import metrics

SECTORS = ("IT", "Banking", "Financial Services", "Energy", "FMCG", "Pharma", "Auto", "Telecom")
ENVIRONMENTS = ("bearish", "bullish", "range_bound", "high_volatility")
ENVIRONMENT_CODES = {name: index for index, name in enumerate(ENVIRONMENTS)}
ENVIRONMENT_SIGNAL = np.array([-0.82, 0.82, 0.02, -0.02])
ENVIRONMENT_VOLATILITY = np.array([0.90, 0.90, 0.70, 1.75])
TRANSITIONS = np.array(
    [
        [0.93, 0.02, 0.03, 0.02],
        [0.02, 0.93, 0.03, 0.02],
        [0.04, 0.04, 0.88, 0.04],
        [0.04, 0.04, 0.08, 0.84],
    ]
)


@dataclass(frozen=True)
class SyntheticSpec:
    """Deterministic specification for the controlled research dataset."""

    seed: int = 20260926
    assets_per_sector: int = 3
    sessions: int = 2200
    start: str = "2018-01-01"
    target_threshold: float = 0.001

    @property
    def assets(self) -> int:
        return len(SECTORS) * self.assets_per_sector


TECHNICAL_FEATURES = [
    "return_1d", "return_3d", "return_5d", "return_10d", "log_return",
    "sma_5", "sma_10", "sma_20", "sma_50", "ema_10", "ema_20", "ema_50",
    "price_to_sma20", "price_to_sma50", "trend_strength", "rsi_14", "macd",
    "macd_signal", "macd_histogram", "momentum_5d", "momentum_10d", "roc_10",
    "rolling_volatility_20", "atr_14", "atr_percentage", "bollinger_width",
]
MARKET_FEATURES = [
    "market_return", "market_momentum", "market_volatility", "sector_return", "sector_momentum",
]
REGIME_FEATURES = [
    "market_environment_code", "regime_duration", "regime_strength", "volatility_regime_code",
]
VOLUME_FEATURES = ["volume_change", "volume_sma_20", "relative_volume", "volume_zscore"]
FEATURE_GROUPS = {
    "technical": TECHNICAL_FEATURES,
    "technical_market": TECHNICAL_FEATURES + MARKET_FEATURES,
    "technical_market_regime": TECHNICAL_FEATURES + MARKET_FEATURES + REGIME_FEATURES,
    "technical_market_regime_volume": TECHNICAL_FEATURES + MARKET_FEATURES + REGIME_FEATURES + VOLUME_FEATURES,
}


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.groupby(level=0).diff()
    gain = delta.clip(lower=0).groupby(level=0).transform(lambda x: x.rolling(window).mean())
    loss = (-delta.clip(upper=0)).groupby(level=0).transform(lambda x: x.rolling(window).mean())
    ratio = gain / loss.replace(0, np.nan)
    result = 100 - 100 / (1 + ratio)
    return result.where(loss.notna(), np.nan).mask((loss == 0) & gain.notna(), 100.0).mask((gain == 0) & loss.notna(), 0.0)


def _asset_definitions(spec: SyntheticSpec, rng: np.random.Generator) -> list[dict[str, Any]]:
    records = []
    for sector_index, sector in enumerate(SECTORS):
        for number in range(spec.assets_per_sector):
            records.append(
                {
                    "symbol": f"SYN{sector_index + 1}{number + 1:02d}",
                    "sector": sector,
                    "base_price": float(rng.uniform(85, 1600)),
                    "base_volume": float(rng.lognormal(14.4, 0.55)),
                    "quality": float(rng.normal(0, 0.12)),
                    "sector_index": sector_index,
                }
            )
    return records


def generate_synthetic_market(spec: SyntheticSpec = SyntheticSpec()) -> pd.DataFrame:
    """Generate a multi-asset OHLCV panel from a persistent latent market process.

    At each session, the next return direction is sampled from a probability influenced
    by the currently observable regime, prior return behaviour, sector behaviour and
    volume/activity persistence.  The hidden innovations and the future sampled return
    are never emitted as model features.
    """
    rng = np.random.default_rng(spec.seed)
    assets = _asset_definitions(spec, rng)
    dates = pd.date_range(spec.start, periods=spec.sessions, freq="B", tz="UTC")
    n_assets, n_days = len(assets), len(dates)
    states = np.empty(n_days, dtype=int)
    states[0] = 2
    for day in range(1, n_days):
        states[day] = rng.choice(len(ENVIRONMENTS), p=TRANSITIONS[states[day - 1]])
    durations = np.ones(n_days, dtype=int)
    for day in range(1, n_days):
        durations[day] = durations[day - 1] + 1 if states[day] == states[day - 1] else 1

    returns = np.zeros((n_days, n_assets))
    closes = np.empty((n_days, n_assets))
    opens = np.empty((n_days, n_assets))
    highs = np.empty((n_days, n_assets))
    lows = np.empty((n_days, n_assets))
    volumes = np.empty((n_days, n_assets))
    closes[0] = [asset["base_price"] for asset in assets]
    opens[0] = closes[0]
    highs[0] = closes[0] * 1.004
    lows[0] = closes[0] * 0.996
    volumes[0] = [asset["base_volume"] for asset in assets]
    sector_previous = np.zeros(len(SECTORS))
    market_previous = 0.0

    for day in range(1, n_days):
        state = states[day]
        current_returns = np.empty(n_assets)
        prior_activity = np.log(volumes[day - 1] / np.maximum(volumes[max(0, day - 20):day].mean(axis=0), 1))
        for index, asset in enumerate(assets):
            short = returns[max(0, day - 3):day, index].mean()
            medium = returns[max(0, day - 10):day, index].mean()
            long = returns[max(0, day - 30):day, index].mean()
            sector_signal = sector_previous[asset["sector_index"]]
            # Observable behaviour drives the probability; independent noise prevents
            # either a threshold rule or a memorisation shortcut from solving the task.
            score = (
                ENVIRONMENT_SIGNAL[state]
                + 48 * short
                + 28 * medium
                - 17 * long
                + 24 * sector_signal
                + 18 * market_previous
                + 0.10 * np.tanh(prior_activity[index])
                + asset["quality"]
                + rng.normal(0, 0.26)
            )
            if state == ENVIRONMENT_CODES["high_volatility"]:
                score *= 0.62
            probability_up = _sigmoid(1.28 * score + 0.06)
            positive = rng.random() < probability_up
            magnitude = (0.0030 + rng.lognormal(-5.15, 0.35)) * ENVIRONMENT_VOLATILITY[state]
            magnitude *= 1 + 0.20 * abs(short) / 0.01
            current_returns[index] = magnitude if positive else -magnitude
        returns[day] = current_returns
        market_previous = float(current_returns.mean())
        for sector_index in range(len(SECTORS)):
            mask = [asset["sector_index"] == sector_index for asset in assets]
            sector_previous[sector_index] = float(current_returns[mask].mean())
        opens[day] = closes[day - 1] * np.exp(rng.normal(0, 0.0015, n_assets) + 0.16 * current_returns)
        closes[day] = closes[day - 1] * np.exp(current_returns)
        ranges = (0.004 + np.abs(current_returns) * 0.58 + rng.uniform(0.001, 0.006, n_assets))
        highs[day] = np.maximum(opens[day], closes[day]) * (1 + ranges)
        lows[day] = np.minimum(opens[day], closes[day]) * (1 - ranges)
        volumes[day] = np.array([asset["base_volume"] for asset in assets]) * np.exp(
            rng.normal(0, 0.28, n_assets) + 13 * np.abs(current_returns) + 0.20 * (state == 3)
        )

    rows = []
    for index, asset in enumerate(assets):
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": dates,
                    "symbol": asset["symbol"],
                    "sector": asset["sector"],
                    "open": opens[:, index],
                    "high": highs[:, index],
                    "low": lows[:, index],
                    "close": closes[:, index],
                    "adjusted_close": closes[:, index],
                    "volume": volumes[:, index].round().astype("int64"),
                    "market_environment": [ENVIRONMENTS[state] for state in states],
                    "market_environment_code": states,
                    "regime_duration": durations,
                    "regime_strength": np.abs(ENVIRONMENT_SIGNAL[states]),
                    "volatility_regime": np.where(states == 3, "high", np.where(states == 2, "low", "moderate")),
                    "volatility_regime_code": np.where(states == 3, 2, np.where(states == 2, 0, 1)),
                }
            )
        )
    return pd.concat(rows, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def build_synthetic_features(raw: pd.DataFrame, horizons: tuple[int, ...] = (1, 3, 5), threshold: float = 0.001) -> pd.DataFrame:
    """Calculate only close-of-session feature values and future-labelled targets."""
    frame = raw.sort_values(["symbol", "timestamp"]).copy()
    grouped = frame.groupby("symbol", group_keys=False)
    frame["return_1d"] = grouped["close"].pct_change()
    frame["log_return"] = grouped["close"].transform(lambda x: np.log(x).diff())
    for window in (3, 5, 10):
        frame[f"return_{window}d"] = grouped["close"].pct_change(window)
        frame[f"momentum_{window}d"] = frame[f"return_{window}d"]
    for window in (5, 10, 20, 50):
        frame[f"sma_{window}"] = grouped["close"].transform(
            lambda x, value=window: x.rolling(value).mean()
        )
    for window in (10, 20, 50):
        frame[f"ema_{window}"] = grouped["close"].transform(
            lambda x, value=window: x.ewm(span=value, adjust=False).mean()
        )
    frame["price_to_sma20"] = frame.close / frame.sma_20 - 1
    frame["price_to_sma50"] = frame.close / frame.sma_50 - 1
    frame["trend_strength"] = frame.sma_5 / frame.sma_20 - frame.sma_20 / frame.sma_50
    indexed = frame.set_index("symbol", drop=False)
    frame["rsi_14"] = _rsi(indexed["close"], 14).to_numpy()
    frame["macd"] = frame.ema_10 - frame.ema_20
    frame["macd_signal"] = frame.groupby("symbol")["macd"].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    frame["macd_histogram"] = frame.macd - frame.macd_signal
    frame["roc_10"] = frame.return_10d
    frame["rolling_volatility_20"] = frame.groupby("symbol")["return_1d"].transform(lambda x: x.rolling(20).std() * np.sqrt(252))
    previous_close = grouped["close"].shift()
    true_range = pd.concat([frame.high - frame.low, (frame.high - previous_close).abs(), (frame.low - previous_close).abs()], axis=1).max(axis=1)
    frame["atr_14"] = true_range.groupby(frame.symbol).transform(lambda x: x.rolling(14).mean())
    frame["atr_percentage"] = frame.atr_14 / frame.close
    frame["bollinger_width"] = 4 * frame.groupby("symbol")["close"].transform(lambda x: x.rolling(20).std()) / frame.sma_20
    frame["volume_change"] = grouped["volume"].pct_change()
    frame["volume_sma_20"] = grouped["volume"].transform(lambda x: x.rolling(20).mean())
    volume_std = grouped["volume"].transform(lambda x: x.rolling(20).std())
    frame["relative_volume"] = frame.volume / frame.volume_sma_20
    frame["volume_zscore"] = (frame.volume - frame.volume_sma_20) / volume_std

    daily = frame.groupby("timestamp", as_index=False).agg(market_return=("return_1d", "mean"))
    daily["market_momentum"] = daily.market_return.rolling(10).sum()
    daily["market_volatility"] = daily.market_return.rolling(20).std() * np.sqrt(252)
    frame = frame.merge(daily, on="timestamp", how="left", validate="many_to_one")
    sector = frame.groupby(["timestamp", "sector"], as_index=False).agg(sector_return=("return_1d", "mean"))
    sector["sector_momentum"] = sector.groupby("sector")["sector_return"].transform(lambda x: x.rolling(10).sum())
    frame = frame.merge(sector, on=["timestamp", "sector"], how="left", validate="many_to_one")
    for horizon in horizons:
        future_close = frame.groupby("symbol")["close"].shift(-horizon)
        available = frame.groupby("symbol")["timestamp"].shift(-horizon)
        frame[f"future_return_{horizon}d"] = future_close / frame.close - 1
        frame[f"target_{horizon}d"] = (frame[f"future_return_{horizon}d"] > threshold).astype("float")
        frame.loc[future_close.isna(), f"target_{horizon}d"] = np.nan
        frame[f"target_available_at_{horizon}d"] = available
    return frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def chronological_panel_split(frame: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by dates, then purge labels whose outcome is unknown at the next period."""
    dates = np.array(sorted(frame.timestamp.drop_duplicates()))
    validation_start = dates[int(len(dates) * 0.70)]
    test_start = dates[int(len(dates) * 0.85)]
    label_time = f"target_available_at_{horizon}d"
    target = f"target_{horizon}d"
    eligible = frame.dropna(subset=[target]).copy()
    train = eligible[(eligible.timestamp < validation_start) & (eligible[label_time] < validation_start)].copy()
    validation = eligible[
        (eligible.timestamp >= validation_start)
        & (eligible.timestamp < test_start)
        & (eligible[label_time] < test_start)
    ].copy()
    test = eligible[eligible.timestamp >= test_start].copy()
    assert train[label_time].max() < validation.timestamp.min()
    assert validation[label_time].max() < test.timestamp.min()
    return train, validation, test


MODEL_CANDIDATES: dict[str, tuple[dict[str, Any], ...]] = {
    "logistic_regression": ({"C": 0.12}, {"C": 0.7}),
    "random_forest": (
        {"n_estimators": 180, "max_depth": 10, "min_samples_leaf": 12, "max_features": 0.7},
        {"n_estimators": 240, "max_depth": 15, "min_samples_leaf": 7, "max_features": 0.8},
    ),
    "xgboost": (
        {"n_estimators": 260, "max_depth": 3, "learning_rate": 0.045, "subsample": 0.82, "colsample_bytree": 0.78, "min_child_weight": 12, "gamma": 0.08, "reg_alpha": 0.08, "reg_lambda": 5.0},
        {"n_estimators": 320, "max_depth": 4, "learning_rate": 0.035, "subsample": 0.78, "colsample_bytree": 0.85, "min_child_weight": 18, "gamma": 0.12, "reg_alpha": 0.12, "reg_lambda": 7.0},
    ),
    "hist_gradient_boosting": (
        {"max_iter": 220, "learning_rate": 0.055, "max_leaf_nodes": 15, "l2_regularization": 2.5},
        {"max_iter": 280, "learning_rate": 0.04, "max_leaf_nodes": 23, "l2_regularization": 4.0},
    ),
}


def make_synthetic_model(name: str, parameters: dict[str, Any]):
    if name == "logistic_regression":
        estimator = LogisticRegression(max_iter=2000, random_state=42, **parameters)
    elif name == "random_forest":
        estimator = RandomForestClassifier(n_jobs=2, random_state=42, **parameters)
    elif name == "xgboost":
        estimator = XGBClassifier(n_jobs=2, random_state=42, eval_metric="logloss", **parameters)
    elif name == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(random_state=42, early_stopping=False, **parameters)
    else:
        raise ValueError(f"Unsupported synthetic model: {name}")
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)


def _fit_select(name: str, train: pd.DataFrame, validation: pd.DataFrame, features: list[str], target: str):
    """Select parameters on validation only; never looks at the future test frame."""
    candidates = []
    for candidate in MODEL_CANDIDATES[name]:
        model = make_synthetic_model(name, candidate)
        model.fit(train[features], train[target].astype(int))
        probability = model.predict_proba(validation[features])[:, 1]
        score = metrics(validation[target], probability)
        candidates.append({"parameters": candidate, "validation": score})
    best = max(candidates, key=lambda item: (item["validation"]["balanced_accuracy"], item["validation"]["roc_auc"]))
    return best, candidates


def _test_result(name: str, selected: dict[str, Any], train_validation: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str):
    model = make_synthetic_model(name, selected["parameters"])
    model.fit(train_validation[features], train_validation[target].astype(int))
    probability = model.predict_proba(test[features])[:, 1]
    return model, probability, metrics(test[target], probability)


def _feature_importance(model, features: list[str], sample: pd.DataFrame) -> list[dict[str, float]]:
    estimator = model[-1]
    values = None
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    if values is None:
        return []
    return [
        {"feature": feature, "importance": float(value)}
        for feature, value in sorted(zip(features, values, strict=True), key=lambda pair: -pair[1])
    ]


def _shap_feature_importance(model, features: list[str], train_validation: pd.DataFrame) -> list[dict[str, float]]:
    """Global SHAP magnitudes from refit training data, never from final-test rows."""
    try:
        import shap

        sample = train_validation.sample(n=min(2000, len(train_validation)), random_state=42)
        background = train_validation.sample(n=min(300, len(train_validation)), random_state=7)
        transformed = model[:-1].transform(sample[features])
        transformed_background = model[:-1].transform(background[features])
        estimator = model[-1]
        if hasattr(estimator, "coef_"):
            values = np.asarray(shap.LinearExplainer(estimator, transformed_background)(transformed).values)
        else:
            values = np.asarray(shap.TreeExplainer(estimator)(transformed).values)
        if values.ndim == 3:
            values = values[:, :, 1]
        importance = np.abs(values).mean(axis=0)
        return [
            {"feature": feature, "mean_absolute_shap": float(value)}
            for feature, value in sorted(zip(features, importance, strict=True), key=lambda pair: -pair[1])
        ]
    except Exception as exc:  # SHAP is optional at runtime; the model result remains valid.
        return [{"unavailable": str(exc)}]


def run_synthetic_experiment(frame: pd.DataFrame) -> tuple[dict[str, Any], Any]:
    """Run an entirely chronological synthetic research experiment.

    Horizon and feature-group comparisons use validation only.  The final test period is
    evaluated once after the selected configuration is frozen.
    """
    horizon_candidates: list[dict[str, Any]] = []
    for horizon in (1, 3, 5):
        train, validation, _ = chronological_panel_split(frame, horizon)
        selected, _ = _fit_select(
            "xgboost", train, validation, FEATURE_GROUPS["technical_market_regime_volume"], f"target_{horizon}d"
        )
        horizon_candidates.append(
            {
                "horizon": horizon,
                "class_distribution": float(validation[f"target_{horizon}d"].mean()),
                "validation": selected["validation"],
            }
        )
    chosen_horizon = max(horizon_candidates, key=lambda item: item["validation"]["balanced_accuracy"])["horizon"]
    train, validation, test = chronological_panel_split(frame, chosen_horizon)
    target = f"target_{chosen_horizon}d"
    feature_group_candidates = []
    for group, features in FEATURE_GROUPS.items():
        selected, _ = _fit_select("xgboost", train, validation, features, target)
        feature_group_candidates.append({"feature_group": group, "validation": selected["validation"]})
    selected_group = max(feature_group_candidates, key=lambda item: item["validation"]["balanced_accuracy"])["feature_group"]
    features = FEATURE_GROUPS[selected_group]

    model_results = []
    selected_models: dict[str, dict[str, Any]] = {}
    combined = pd.concat([train, validation], ignore_index=True)
    for name in MODEL_CANDIDATES:
        selected, candidates = _fit_select(name, train, validation, features, target)
        model, probability, score = _test_result(name, selected, combined, test, features, target)
        row = {
            "model": name,
            "parameters": selected["parameters"],
            "validation": selected["validation"],
            "test": score,
            "candidate_validation": candidates,
        }
        model_results.append(row)
        selected_models[name] = {"model": model, "probability": probability, "record": row}
    winning_name = max(model_results, key=lambda item: (item["validation"]["balanced_accuracy"], item["validation"]["roc_auc"]))["model"]
    winner = selected_models[winning_name]
    predictions = test[["timestamp", "symbol", "sector", "market_environment", target]].copy()
    predictions["probability_up"] = winner["probability"]
    predictions["predicted_up"] = (winner["probability"] >= 0.5).astype(int)
    asset_results = []
    for symbol, group in predictions.groupby("symbol"):
        score = metrics(group[target], group.probability_up)
        asset_results.append({"symbol": symbol, "sector": group.sector.iloc[0], "n": len(group), "metrics": score})
    environment_results = []
    for environment, group in predictions.groupby("market_environment"):
        score = metrics(group[target], group.probability_up)
        environment_results.append({"environment": environment, "n": len(group), "metrics": score})
    trainval_label_time = f"target_available_at_{chosen_horizon}d"
    assert combined[trainval_label_time].max() < test.timestamp.min()
    dataset_info = {
        "dataset_type": "controlled_synthetic_market_like",
        "disclaimer": "Controlled synthetic market-like data is used only to evaluate ML methodology. Its scores do not measure or imply real Indian-market forecasting accuracy.",
        "observations": int(len(frame.dropna(subset=[target]))),
        "assets": int(frame.symbol.nunique()),
        "sectors": int(frame.sector.nunique()),
        "date_range": {"start": str(frame.timestamp.min().date()), "end": str(frame.timestamp.max().date())},
        "class_distribution": {
            "up": float(frame.dropna(subset=[target])[target].mean()),
            "down": float(1 - frame.dropna(subset=[target])[target].mean()),
        },
        "majority_accuracy": float(max(frame.dropna(subset=[target])[target].mean(), 1 - frame.dropna(subset=[target])[target].mean())),
    }
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": dataset_info,
        "methodology": "Date-based 70% train / 15% validation / 15% final test split. Labels are purged from each preceding period until their target date is known. Hyperparameters, horizon and feature group are selected only by validation balanced accuracy then ROC-AUC. The final test is evaluated after selection.",
        "target": {"horizon_sessions": chosen_horizon, "threshold": 0.001, "definition": "UP when the selected forward synthetic close-to-close return exceeds +0.10%; otherwise DOWN."},
        "horizon_experiments": horizon_candidates,
        "feature_group_experiments": feature_group_candidates,
        "selected": {"model": winning_name, "feature_group": selected_group, "features": features, "validation": winner["record"]["validation"], "test": winner["record"]["test"], "parameters": winner["record"]["parameters"]},
        "model_results": model_results,
        "asset_results": asset_results,
        "environment_results": environment_results,
        "feature_importance": _feature_importance(winner["model"], features, combined),
        "shap_feature_importance": _shap_feature_importance(winner["model"], features, combined),
        "split": {"train_end": str(train.timestamp.max().date()), "validation_start": str(validation.timestamp.min().date()), "validation_end": str(validation.timestamp.max().date()), "test_start": str(test.timestamp.min().date()), "test_end": str(test.timestamp.max().date())},
    }
    artifact = {
        "dataset_type": dataset_info["dataset_type"],
        "disclaimer": dataset_info["disclaimer"],
        "model": winner["model"],
        "features": features,
        "target": report["target"],
        "selected": report["selected"],
        "training_timestamp": report["created_at"],
        "split": report["split"],
    }
    return report, artifact


def save_synthetic_outputs(frame: pd.DataFrame, report: dict[str, Any], artifact: dict[str, Any]) -> dict[str, str]:
    """Persist ignored data/model files plus the small, inspectable result report."""
    data_path = Path("data/processed/synthetic_market_research.csv.gz")
    report_path = Path("docs/synthetic_research_results.json")
    artifact_path = Path("models/synthetic-research-selected.joblib")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data_path, index=False, compression="gzip")
    import json

    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    joblib.dump(artifact, artifact_path)
    return {"data": str(data_path), "report": str(report_path), "artifact": str(artifact_path)}


def spec_metadata(spec: SyntheticSpec) -> dict[str, Any]:
    return asdict(spec)
