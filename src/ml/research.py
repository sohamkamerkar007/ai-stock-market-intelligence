"""Reproducible model selection with purged chronological development folds.

The final 15% is never used for choosing configuration, calibration or threshold.
Previously published holdouts are explicitly retrospective, not virgin test data.
"""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features.pipeline import expanded_features
from src.regimes.discovery import causal_regime_labels

CONFIGURATIONS = {
    "logistic_regularized": {"model": "logistic_regression", "C": 0.05},
    "logistic_balanced": {"model": "logistic_regression", "C": 0.2, "class_weight": "balanced"},
    "forest_regularized": {"model": "random_forest", "max_depth": 5, "min_samples_leaf": 25},
    "xgb_shallow": {
        "model": "xgboost",
        "max_depth": 2,
        "min_child_weight": 20,
        "reg_lambda": 10,
        "reg_alpha": 1,
    },
    "xgb_medium": {
        "model": "xgboost",
        "max_depth": 3,
        "min_child_weight": 10,
        "reg_lambda": 5,
        "gamma": 0.2,
    },
    "hist_regularized": {
        "model": "hist_gradient_boosting",
        "max_leaf_nodes": 7,
        "l2_regularization": 10,
    },
    "extra_trees": {"model": "extra_trees", "max_depth": 6, "min_samples_leaf": 20},
}


def make_model(config: str):
    parameters = CONFIGURATIONS[config].copy()
    name = parameters.pop("model")
    if name == "logistic_regression":
        estimator = LogisticRegression(max_iter=2000, random_state=42, **parameters)
    elif name == "xgboost":
        estimator = XGBClassifier(
            n_estimators=120,
            learning_rate=0.025,
            subsample=0.8,
            colsample_bytree=0.7,
            random_state=42,
            n_jobs=2,
            eval_metric="logloss",
            **parameters,
        )
    elif name == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.04, early_stopping=False, random_state=42, **parameters
        )
    else:
        cls = RandomForestClassifier if name == "random_forest" else ExtraTreesClassifier
        estimator = cls(n_estimators=120, max_features=0.7, n_jobs=2, random_state=42, **parameters)
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), estimator
    )


def metrics(y, probability):
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    pred = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) > 1 else None,
        "mcc": float(matthews_corrcoef(y, pred)),
        "brier": float(brier_score_loss(y, probability)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "n": len(y),
        "actual_up_fraction": float(y.mean()),
        "mean_probability": float(probability.mean()),
        "majority_accuracy": float(max(y.mean(), 1 - y.mean())),
    }


def context_frames(price_frames: dict[str, pd.DataFrame]):
    """Join by actual session date; never align separate assets by row number."""
    context = None
    for symbol in ["NIFTY50", "BANKNIFTY", "SENSEX"]:
        if symbol not in price_frames:
            continue
        frame, _ = expanded_features(price_frames[symbol])
        cols = ["return_1d", "return_5d", "return_20d", "volatility_20", "sma_ratio_20"]
        part = frame[["timestamp", *cols]].rename(columns={c: f"{symbol}_{c}" for c in cols})
        context = (
            part
            if context is None
            else context.merge(part, on="timestamp", how="outer", validate="one_to_one")
        )
    nifty, _ = expanded_features(price_frames["NIFTY50"])
    nifty["regime_state"] = causal_regime_labels(nifty)
    state = nifty.regime_state
    nifty["regime_duration"] = (
        state.groupby(state.ne(state.shift()).fillna(True).cumsum()).cumcount() + 1
    )
    nifty["regime_transition"] = state.ne(state.shift()).astype(float)
    for i in range(3):
        nifty[f"regime_{i}"] = (state == i).astype(float)
    return context, nifty[
        [
            "timestamp",
            "regime_state",
            "regime_duration",
            "regime_transition",
            "regime_0",
            "regime_1",
            "regime_2",
        ]
    ]


def dataset(prices, market, regimes):
    frame, technical = expanded_features(prices)
    frame = frame.merge(market, on="timestamp", how="left", validate="one_to_one")
    frame = frame.merge(regimes, on="timestamp", how="left", validate="one_to_one")
    frame["relative_strength_nifty"] = frame.return_5d - frame.NIFTY50_return_5d
    market_columns = [c for c in market if c != "timestamp"] + ["relative_strength_nifty"]
    regime_columns = ["regime_0", "regime_1", "regime_2", "regime_duration", "regime_transition"]
    # Common start makes feature-set comparisons use exactly the same sessions.
    frame = frame[frame.regime_state.notna() & frame.sma_ratio_100.notna()].reset_index(drop=True)
    return frame, {
        "technical": technical,
        "technical_market": technical + market_columns,
        "technical_environment": technical + regime_columns,
        "technical_market_environment": technical + market_columns + regime_columns,
    }


def fit_calibrated(train, columns, config, horizon, calibration):
    model = make_model(config)
    calibrator = None
    if calibration:
        split = int(len(train) * 0.8)
        fitting = train.iloc[: max(1, split - horizon)]
        cal = train.iloc[split:]
        model.fit(fitting[columns], fitting.target_up.astype(int))
        raw = model.predict_proba(cal[columns])[:, 1]
        calibrator = LogisticRegression(C=1, random_state=42).fit(
            np.log(np.clip(raw, 1e-6, 1 - 1e-6) / (1 - np.clip(raw, 1e-6, 1 - 1e-6))).reshape(
                -1, 1
            ),
            cal.target_up.astype(int),
        )
    else:
        model.fit(train[columns], train.target_up.astype(int))
    return model, calibrator


def predict(model, calibrator, x):
    p = model.predict_proba(x)[:, 1]
    if calibrator is not None:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        p = calibrator.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]
    return p


def evaluate_configuration(frame, columns, config, horizon, calibration=False):
    # The last 15% remains sealed throughout this function.
    development = frame.iloc[: int(len(frame) * 0.85) - horizon]
    fold_results, ys, ps = [], [], []
    for train_idx, val_idx in TimeSeriesSplit(n_splits=3, gap=horizon).split(development):
        train, val = development.iloc[train_idx], development.iloc[val_idx]
        assert train.label_available_at.max() < val.timestamp.min()
        model, calibrator = fit_calibrated(train, columns, config, horizon, calibration)
        p = predict(model, calibrator, val[columns])
        score = metrics(val.target_up, p)
        score.update(
            {
                "train_end": str(train.timestamp.max()),
                "label_end": str(train.label_available_at.max()),
                "validation_start": str(val.timestamp.min()),
                "validation_end": str(val.timestamp.max()),
            }
        )
        fold_results.append(score)
        ys.extend(val.target_up.astype(int))
        ps.extend(p)
    summary = metrics(ys, ps)
    summary["fold_accuracy_mean"] = float(np.mean([f["accuracy"] for f in fold_results]))
    summary["fold_accuracy_std"] = float(np.std([f["accuracy"] for f in fold_results]))
    return {
        "configuration": config,
        "model": CONFIGURATIONS[config]["model"],
        "horizon": horizon,
        "calibration": calibration,
        "validation": summary,
        "folds": fold_results,
    }


def final_evaluation(frame, columns, selected):
    boundary = int(len(frame) * 0.85)
    test = frame.iloc[boundary:]
    train = frame.iloc[:boundary]
    train = train[train.label_available_at < test.timestamp.min()]
    model, calibrator = fit_calibrated(
        train, columns, selected["configuration"], selected["horizon"], selected["calibration"]
    )
    probability = predict(model, calibrator, test[columns])
    observations = test[["timestamp", "target_up", "regime_state"]].copy()
    observations["probability"] = probability
    observations["actual_return"] = test.future_return
    regression = make_pipeline(
        SimpleImputer(keep_empty_features=True), StandardScaler(), Ridge(alpha=100)
    )
    regression.fit(train[columns], train.future_return)
    predicted_return = regression.predict(test[columns])
    regression_metrics = {
        "mae": float(mean_absolute_error(test.future_return, predicted_return)),
        "rmse": float(np.sqrt(mean_squared_error(test.future_return, predicted_return))),
        "r2": float(r2_score(test.future_return, predicted_return)),
        "zero_return_mae": float(np.abs(test.future_return).mean()),
    }
    score = metrics(test.target_up, probability)
    score["regression"] = regression_metrics
    return score, observations, model, calibrator, regression


def snapshot_hash(frame):
    return hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=False).values.tobytes()
    ).hexdigest()


def save_report(report, destination=Path("docs/experiment_results.json")):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, allow_nan=False, default=str), encoding="utf-8"
    )


def save_bundle(bundle, symbol):
    path = Path("models") / f"selected-{symbol}.joblib"
    joblib.dump(bundle, path)
    return str(path)
