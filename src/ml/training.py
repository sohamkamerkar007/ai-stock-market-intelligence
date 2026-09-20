import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


@dataclass
class TrainingResult:
    run_id: str
    model_name: str
    metrics: dict[str, Any]
    split_dates: dict[str, str]
    artifact_path: str


def chronological_split(
    frame: pd.DataFrame, train_ratio: float = 0.7, validation_ratio: float = 0.15
):
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    a, b = int(len(ordered) * train_ratio), int(len(ordered) * (train_ratio + validation_ratio))
    if a < 30 or b <= a or len(ordered) - b < 10:
        raise ValueError("Insufficient observations for chronological train/validation/test split")
    return ordered.iloc[:a], ordered.iloc[a:b], ordered.iloc[b:]


def model_catalog(seed: int = 42) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=2,
        ),
    }


def evaluate_classifier(model: object, x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    pred = model.predict(x)
    probability = model.predict_proba(x)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)) if y.nunique() > 1 else None,
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def train_classifier(
    frame: pd.DataFrame,
    features: list[str],
    name: str,
    output_dir: Path = Path("models"),
    seed: int = 42,
) -> TrainingResult:
    usable = frame.dropna(subset=["target_up"]).copy()
    train, validation, test = chronological_split(usable)
    if name not in model_catalog(seed):
        raise ValueError(f"Unknown model: {name}")
    scaler = StandardScaler() if name == "logistic_regression" else "passthrough"
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", scaler)]),
                features,
            )
        ]
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model_catalog(seed)[name])])
    pipeline.fit(
        pd.concat([train, validation])[features],
        pd.concat([train, validation])["target_up"].astype(int),
    )
    metrics = evaluate_classifier(pipeline, test[features], test["target_up"].astype(int))
    run_id = uuid.uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}-{run_id}.joblib"
    joblib.dump(
        {
            "pipeline": pipeline,
            "features": features,
            "trained_until": validation["timestamp"].max(),
        },
        path,
    )
    metadata = {
        "run_id": run_id,
        "model": name,
        "metrics": metrics,
        "features": features,
        "split": {
            "train_end": str(train["timestamp"].max()),
            "validation_end": str(validation["timestamp"].max()),
            "test_end": str(test["timestamp"].max()),
        },
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return TrainingResult(run_id, name, metrics, metadata["split"], str(path))


def walk_forward_probabilities(
    frame: pd.DataFrame, features: list[str], min_train: int = 252, retrain_every: int = 20
) -> pd.Series:
    ordered = frame.sort_values("timestamp").reset_index()
    output = pd.Series(index=frame.index, dtype=float)
    pipeline = None
    for i in range(min_train, len(ordered)):
        if pipeline is None or (i - min_train) % retrain_every == 0:
            train = ordered.iloc[:i].dropna(subset=["target_up"])
            pipeline = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
                ]
            )
            pipeline.fit(train[features], train["target_up"].astype(int))
        output.loc[ordered.loc[i, "index"]] = pipeline.predict_proba(ordered.loc[[i], features])[
            :, 1
        ][0]
    return output
