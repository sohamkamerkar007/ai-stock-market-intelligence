import argparse

from backend.app.database import SessionLocal
from backend.app.models import ModelRun
from scripts.common import load_asset_frame
from src.features import FEATURE_COLUMNS, build_features, build_target
from src.ml.training import train_classifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="NIFTY50")
    parser.add_argument(
        "--models", nargs="+", default=["logistic_regression", "random_forest", "xgboost"]
    )
    args = parser.parse_args()
    with SessionLocal() as db:
        _, prices = load_asset_frame(db, args.symbol)
        dataset = build_target(build_features(prices)).dropna(subset=FEATURE_COLUMNS)
        for name in args.models:
            result = train_classifier(dataset, FEATURE_COLUMNS, name)
            db.add(
                ModelRun(
                    run_id=result.run_id,
                    task="next_day_direction",
                    model_name=name,
                    feature_set="technical_v1",
                    trained_from=dataset.timestamp.min().date(),
                    trained_to=dataset.timestamp.max().date(),
                    metrics=result.metrics,
                    parameters={"seed": 42, "validation": "chronological_70_15_15"},
                    artifact_path=result.artifact_path,
                )
            )
            db.commit()
            print(name, result.metrics)


if __name__ == "__main__":
    main()
