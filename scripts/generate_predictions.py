"""Train asset-specific tree models and persist current predictions plus SHAP explanations."""

import argparse
from datetime import UTC, datetime

import joblib
import pandas as pd
from sqlalchemy import desc, select

from backend.app.database import SessionLocal
from backend.app.models import (
    Asset,
    MarketRegime,
    ModelRun,
    Prediction,
    PredictionExplanation,
)
from scripts.common import load_asset_frame
from src.explainability.shap_explanations import explain_contributions, tree_shap
from src.features import FEATURE_COLUMNS, build_features, build_target
from src.ml.training import train_classifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="*")
    args = parser.parse_args()
    with SessionLocal() as db:
        symbols = args.symbols or list(
            db.scalars(select(Asset.symbol).where(Asset.active.is_(True))).all()
        )
        regime = db.scalar(select(MarketRegime).order_by(desc(MarketRegime.timestamp)).limit(1))
        for symbol in symbols:
            try:
                asset, prices = load_asset_frame(db, symbol)
                features = build_features(prices)
                dataset = build_target(features).dropna(subset=FEATURE_COLUMNS)
                result = train_classifier(dataset, FEATURE_COLUMNS, "xgboost")
                run = ModelRun(
                    run_id=result.run_id,
                    task="next_day_direction",
                    model_name="xgboost",
                    feature_set=f"technical_v1:{symbol}",
                    trained_from=dataset.timestamp.min().date(),
                    trained_to=dataset.timestamp.max().date(),
                    metrics=result.metrics,
                    parameters={"seed": 42, "asset": symbol, "no_shuffle": True},
                    artifact_path=result.artifact_path,
                )
                db.add(run)
                db.flush()
                bundle = joblib.load(result.artifact_path)
                latest = features.dropna(subset=FEATURE_COLUMNS).iloc[[-1]]
                pipeline = bundle["pipeline"]
                probability = float(pipeline.predict_proba(latest[FEATURE_COLUMNS])[:, 1][0])
                prediction = Prediction(
                    asset_id=asset.id,
                    model_run_id=run.id,
                    prediction_for=(
                        pd.Timestamp(latest.timestamp.iloc[0]) + pd.offsets.BDay(1)
                    ).date(),
                    generated_at=datetime.now(UTC),
                    direction="UP" if probability >= 0.5 else "DOWN",
                    probability_up=probability,
                    expected_return=None,
                    regime=regime.label if regime else None,
                )
                db.add(prediction)
                db.flush()
                transformed = pipeline.named_steps["preprocess"].transform(latest[FEATURE_COLUMNS])
                attribution = tree_shap(pipeline.named_steps["model"], transformed, FEATURE_COLUMNS)
                summary = explain_contributions(
                    attribution["features"],
                    attribution["values"],
                    probability,
                )
                db.add(
                    PredictionExplanation(
                        prediction_id=prediction.id,
                        contributions=dict(
                            zip(attribution["features"], attribution["values"], strict=True)
                        ),
                        summary=summary,
                    )
                )
                db.commit()
                print(symbol, prediction.direction, round(probability, 4), result.metrics)
            except Exception as exc:
                db.rollback()
                print("SKIP", symbol, str(exc))


if __name__ == "__main__":
    main()
