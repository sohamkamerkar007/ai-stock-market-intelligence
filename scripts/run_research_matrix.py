"""Run bounded, purged experiments and publish the pre-holdout-selected one-day model."""

import argparse
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from sqlalchemy import select
from threadpoolctl import threadpool_limits

from backend.app.database import SessionLocal
from backend.app.models import Asset, ModelRun, NewsArticle, Prediction, PredictionExplanation
from scripts.common import load_asset_frame
from src.features.pipeline import build_target
from src.ml.research import (
    CONFIGURATIONS,
    context_frames,
    dataset,
    evaluate_configuration,
    final_evaluation,
    fit_calibrated,
    metrics,
    predict,
    save_bundle,
    save_report,
    snapshot_hash,
)


def environment_label(row: pd.Series) -> str:
    """Describe only contemporaneous, observed market characteristics."""
    recent_return = float(row.get("NIFTY50_return_20d", 0.0))
    volatility = float(row.get("NIFTY50_volatility_20", 0.0))
    momentum = float(row.get("momentum_10", 0.0))
    direction = "Positive trend" if recent_return > 0.01 else "Weak market" if recent_return < -0.01 else "Range-bound"
    risk = "high volatility" if volatility >= 0.20 else "moderate volatility" if volatility >= 0.12 else "lower volatility"
    impulse = "positive momentum" if momentum > 0 else "weak momentum"
    return f"{direction} · {risk} · {impulse}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symbols", nargs="*", help="Asset evaluation subset; default all active assets"
    )
    parser.add_argument(
        "--publish", action="store_true", help="Persist new predictions and production bundles"
    )
    args = parser.parse_args()
    with SessionLocal() as db, threadpool_limits(limits=2):
        assets = list(db.scalars(select(Asset).where(Asset.active.is_(True))))
        prices = {a.symbol: load_asset_frame(db, a.symbol)[1] for a in assets}
        market, environments = context_frames(prices)
        nifty, feature_sets = dataset(prices["NIFTY50"], market, environments)
        report = {
            "created_at": datetime.now(UTC).isoformat(),
            "seed": 42,
            "methodology": "Three expanding development folds with horizon-length purge; select by development balanced accuracy then Brier score; final 15% retrospective holdout. One-day remains primary; 3/5-day exploratory.",
            "holdout_warning": "The legacy holdout has already been seen. Results are retrospective and require a future prospective evaluation. No settings selected using holdout scores.",
            "configurations": CONFIGURATIONS,
            "feature_sets": feature_sets,
            "data": {
                s: {
                    "rows": len(p),
                    "start": str(p.timestamp.min()),
                    "end": str(p.timestamp.max()),
                    "duplicates": int(p.timestamp.duplicated().sum()),
                    "sha256": snapshot_hash(p),
                }
                for s, p in prices.items()
            },
            "news_status": "No historical news rows available; sentiment experiment skipped"
            if db.scalar(select(NewsArticle.id).limit(1)) is None
            else "News available; requires publication-time coverage validation before inclusion",
            "experiments": [],
            "assets": [],
        }
        for horizon in [1, 3, 5]:
            frame = build_target(nifty, horizon).dropna(subset=["target_up"])
            for feature_set, columns in feature_sets.items():
                for config in CONFIGURATIONS:
                    result = evaluate_configuration(frame, columns, config, horizon)
                    result["feature_set"] = feature_set
                    report["experiments"].append(result)
                    print(
                        horizon,
                        feature_set,
                        config,
                        round(result["validation"]["balanced_accuracy"], 4),
                        flush=True,
                    )
            # Calibration is a development-only comparison against the horizon winner.
            candidates = [r for r in report["experiments"] if r["horizon"] == horizon]
            winner = max(
                candidates,
                key=lambda r: (r["validation"]["balanced_accuracy"], -r["validation"]["brier"]),
            )
            calibrated = evaluate_configuration(
                frame, feature_sets[winner["feature_set"]], winner["configuration"], horizon, True
            )
            calibrated["feature_set"] = winner["feature_set"]
            report["experiments"].append(calibrated)
            save_report(report)
        selected = max(
            [r for r in report["experiments"] if r["horizon"] == 1],
            key=lambda r: (r["validation"]["balanced_accuracy"], -r["validation"]["brier"]),
        )
        # Freeze selection before opening any final holdout.
        report["selected"] = {
            k: selected[k]
            for k in ["configuration", "model", "horizon", "feature_set", "calibration"]
        }
        save_report(report)
        for result in report["experiments"]:
            frame = build_target(nifty, result["horizon"]).dropna(subset=["target_up"])
            result["holdout"], _, _, _, _ = final_evaluation(
                frame, feature_sets[result["feature_set"]], result
            )
        selected_metrics = next(
            r
            for r in report["experiments"]
            if all(r[k] == v for k, v in report["selected"].items())
        )
        report["selected"]["holdout"] = selected_metrics["holdout"]
        all_observations = []
        for asset in assets:
            if args.symbols and asset.symbol not in args.symbols:
                continue
            frame, sets = dataset(prices[asset.symbol], market, environments)
            latest = frame.iloc[[-1]]
            labeled = build_target(frame, 1).dropna(subset=["target_up"])
            columns = sets[selected["feature_set"]]
            score, observations, model, calibrator, regression = final_evaluation(
                labeled, columns, selected
            )
            observations["symbol"] = asset.symbol
            all_observations.append(observations)
            report["assets"].append(
                {
                    "symbol": asset.symbol,
                    "metrics": score,
                    "start": str(observations.timestamp.min()),
                    "end": str(observations.timestamp.max()),
                }
            )
            print("ASSET", asset.symbol, score, flush=True)
            if args.publish:
                # Deployment refit is separate from the frozen evaluation artifact.
                model, calibrator = fit_calibrated(
                    labeled, columns, selected["configuration"], 1, selected["calibration"]
                )
                regression.fit(labeled[columns], labeled.future_return)
                bundle = {
                    "pipeline": model,
                    "calibrator": calibrator,
                    "regression": regression,
                    "features": columns,
                    "selection": report["selected"],
                    "trained_until": str(labeled.label_available_at.max()),
                }
                path = save_bundle(bundle, asset.symbol)
                run = ModelRun(
                    run_id=uuid4().hex,
                    task="direction_research_v2",
                    model_name=selected["model"],
                    feature_set=f"v2:{asset.symbol}",
                    trained_from=labeled.timestamp.min().date(),
                    trained_to=labeled.timestamp.max().date(),
                    metrics=score,
                    parameters={
                        **report["selected"],
                        "asset": asset.symbol,
                        "validation": report["methodology"],
                    },
                    artifact_path=path,
                )
                db.add(run)
                db.flush()
                probability = float(predict(model, calibrator, latest[columns])[0])
                state = int(latest.regime_state.iloc[0])
                prediction = Prediction(
                    asset_id=asset.id,
                    model_run_id=run.id,
                    prediction_for=(
                        pd.Timestamp(latest.timestamp.iloc[0]) + pd.offsets.BDay(1)
                    ).date(),
                    direction="UP" if probability >= 0.5 else "DOWN",
                    probability_up=probability,
                    expected_return=float(regression.predict(latest[columns])[0]),
                    regime=f"{environment_label(latest.iloc[0])}; causal state ID {state}",
                )
                db.add(prediction)
                db.flush()
                try:
                    import shap

                    transformed = model[:-1].transform(latest[columns])
                    estimator = model[-1]
                    if hasattr(estimator, "coef_"):
                        background = model[:-1].transform(labeled[columns].tail(252))
                        explanation = shap.LinearExplainer(estimator, background)(transformed)
                    else:
                        explanation = shap.TreeExplainer(estimator)(transformed)
                    values = np.asarray(explanation.values)
                    if values.ndim == 3:
                        values = values[:, :, 1]
                    contributions = dict(
                        zip(columns, values[0].astype(float).tolist(), strict=True)
                    )
                    db.add(
                        PredictionExplanation(
                            prediction_id=prediction.id,
                            contributions=contributions,
                            summary="SHAP contributions explain the underlying direction model. Positive values support UP; negative values support DOWN. Contributions are not causal effects. If calibrated, the probability mapping is applied after these model contributions.",
                        )
                    )
                    if asset.symbol == "NIFTY50":
                        report["feature_importance"] = sorted(
                            [
                                {"feature": k, "absolute_contribution": abs(v)}
                                for k, v in contributions.items()
                            ],
                            key=lambda x: -x["absolute_contribution"],
                        )
                except (ValueError, TypeError, NotImplementedError) as exc:
                    report.setdefault("explanation_errors", []).append(
                        {"symbol": asset.symbol, "error": str(exc)}
                    )
                db.commit()
        combined = pd.concat(all_observations, ignore_index=True)
        Path("models").mkdir(exist_ok=True)
        combined.to_csv("models/research_predictions.csv", index=False)
        report["environment_performance"] = [
            {"state": int(state), **metrics(g.target_up, g.probability)}
            for state, g in combined.groupby("regime_state")
        ]
        report["probability_bins"] = []
        for lower in np.arange(0, 1, 0.1):
            group = combined[
                (combined.probability >= lower) & (combined.probability < lower + 0.1000001)
            ]
            if len(group):
                report["probability_bins"].append(
                    {
                        "lower": round(float(lower), 1),
                        "count": len(group),
                        "mean_probability": float(group.probability.mean()),
                        "observed_up_rate": float(group.target_up.mean()),
                    }
                )
        save_report(report)
        print("SELECTED", report["selected"], flush=True)


if __name__ == "__main__":
    main()
