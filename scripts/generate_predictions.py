"""Train asset-specific tree models and persist current predictions plus SHAP explanations."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

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


def generate_selected(db, symbols):
    """Inference only: consume frozen selected artifacts without replacing the model."""
    import numpy as np
    import shap

    from src.ml.research import context_frames, dataset, predict

    all_symbols = list(db.scalars(select(Asset.symbol).where(Asset.active.is_(True))))
    frames = {s: load_asset_frame(db, s)[1] for s in all_symbols}
    market, environments = context_frames(frames)
    regime = db.scalar(select(MarketRegime).join(Asset).where(
        Asset.symbol == "NIFTY50", MarketRegime.algorithm == "hmm"
    ).order_by(desc(MarketRegime.timestamp)))
    for symbol in symbols:
        path = Path("models") / f"selected-{symbol}.joblib"
        if not path.exists():
            raise ValueError(f"Selected model missing for {symbol}; run scripts.run_research_matrix --publish")
        bundle = joblib.load(path)
        frame, _ = dataset(frames[symbol], market, environments)
        latest = frame.iloc[[-1]]
        columns, model = bundle["features"], bundle["pipeline"]
        run = db.scalar(select(ModelRun).where(ModelRun.artifact_path == str(path)).order_by(desc(ModelRun.created_at)))
        if run is None:
            raise ValueError(f"No model-run provenance for {symbol}; publish the research matrix first")
        asset = db.scalar(select(Asset).where(Asset.symbol == symbol))
        probability = float(predict(model, bundle["calibrator"], latest[columns])[0])
        output = Prediction(asset_id=asset.id, model_run_id=run.id,
            prediction_for=(pd.Timestamp(latest.timestamp.iloc[0])+pd.offsets.BDay(1)).date(),
            direction="UP" if probability >= .5 else "DOWN", probability_up=probability,
            expected_return=float(bundle["regression"].predict(latest[columns])[0]),
            regime=regime.label if regime else None)
        db.add(output)
        db.flush()
        x = model[:-1].transform(latest[columns])
        estimator = model[-1]
        explainer = shap.LinearExplainer(estimator, model[:-1].transform(frame[columns].tail(252))) if hasattr(estimator, "coef_") else shap.TreeExplainer(estimator)
        values = np.asarray(explainer(x).values)
        if values.ndim == 3:
            values = values[:, :, 1]
        db.add(PredictionExplanation(prediction_id=output.id,
               contributions=dict(zip(columns, values[0].astype(float).tolist(), strict=True)),
               summary="Positive SHAP contributions support upward movement; negative contributions support downward movement. These explain the fitted model, not causal effects."))
        inputs = dict((run.parameters or {}).get("prediction_inputs", {}))
        inputs[str(output.id)] = {c: float(latest[c].iloc[0]) if pd.notna(latest[c].iloc[0]) else None for c in columns}
        run.parameters = {**run.parameters, "prediction_inputs": inputs}
        db.commit()
        print(symbol, output.direction, probability, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--legacy-train", action="store_true", help="Explicitly retrain the legacy XGBoost pipeline instead of using selected artifacts")
    args = parser.parse_args()
    with SessionLocal() as db:
        symbols = args.symbols or list(
            db.scalars(select(Asset.symbol).where(Asset.active.is_(True))).all()
        )
        if not args.legacy_train:
            generate_selected(db, symbols)
            return
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
