"""Train the two controlled direction models after a fixed development selection."""

import json
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd

from src.ml.final_prediction import FINAL_FEATURES, ROOT, development_split, load_research_panel
from src.ml.research import metrics
from src.ml.synthetic_research import MODEL_CANDIDATES, make_synthetic_model

HORIZON = 3
FEATURES = FINAL_FEATURES
# Frozen after development-period validation; no final-test tuning.
PARAMETERS = {
    "logistic_regression": MODEL_CANDIDATES["logistic_regression"][1],
    "xgboost": MODEL_CANDIDATES["xgboost"][1],
}


def main() -> None:
    frame = load_research_panel()
    train, validation, test = development_split(frame, HORIZON)
    target = f"target_{HORIZON}d"
    trainval = pd.concat([train, validation], ignore_index=True)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_type": "controlled_synthetic_market_like",
        "disclaimer": "Controlled synthetic data only; these metrics do not measure Indian-stock prediction accuracy.",
        "target": "UP if the next 3-session synthetic close-to-close return exceeds +0.10%; otherwise DOWN",
        "split": {
            "train_end": str(train.timestamp.max().date()),
            "validation_start": str(validation.timestamp.min().date()),
            "validation_end": str(validation.timestamp.max().date()),
            "test_start": str(test.timestamp.min().date()),
            "test_end": str(test.timestamp.max().date()),
            "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
        },
        "class_distribution": {"train_up": float(train[target].mean()), "validation_up": float(validation[target].mean()), "test_up": float(test[target].mean())},
        "feature_set": "Technical + market context + current observable simulated state + volume",
        "features": FEATURES,
        "baseline": {
            "name": "Training-majority constant classifier",
            "predicted_class": int(train[target].mean() >= .5),
            "test": metrics(test[target], np.full(len(test), float(train[target].mean() >= .5))),
        },
        "models": {},
    }
    output = ROOT / "models/final"
    output.mkdir(parents=True, exist_ok=True)
    for name, parameters in PARAMETERS.items():
        development_model = make_synthetic_model(name, parameters)
        development_model.fit(train[FEATURES], train[target].astype(int))
        validation_probability = development_model.predict_proba(validation[FEATURES])[:, 1]
        validation_metrics = metrics(validation[target], validation_probability)
        model = make_synthetic_model(name, parameters)
        model.fit(trainval[FEATURES], trainval[target].astype(int))
        test_probability = model.predict_proba(test[FEATURES])[:, 1]
        test_metrics = metrics(test[target], test_probability)
        report["models"][name] = {"parameters": parameters, "validation": validation_metrics, "test": test_metrics}
        joblib.dump({"model": model, "features": FEATURES, "horizon": HORIZON, "dataset_type": report["dataset_type"], "report": report["models"][name], "train_end": report["split"]["validation_end"]}, output / f"synthetic_{name}.joblib")
        print(name, "validation", validation_metrics["accuracy"], "unseen test", test_metrics["accuracy"], flush=True)
    path = ROOT / "docs/final_classification_results.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("Saved", path)


if __name__ == "__main__":
    main()
