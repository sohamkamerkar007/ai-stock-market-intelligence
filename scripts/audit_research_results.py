"""Generate comparable legacy results, feature diagnostics and corrected backtests."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import desc, select
from threadpoolctl import threadpool_limits

from backend.app.database import SessionLocal
from backend.app.models import ModelRun
from backend.app.services import execute_backtest
from scripts.common import load_asset_frame
from src.features.pipeline import FEATURE_COLUMNS, build_features, build_target, expanded_features
from src.ml.research import metrics, save_report


def main():
    path = Path("docs/experiment_results.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    observed = pd.read_csv("models/research_predictions.csv")
    nifty = observed[observed.symbol == "NIFTY50"]
    with SessionLocal() as db, threadpool_limits(limits=2):
        _, prices = load_asset_frame(db, "NIFTY50")
        frame = build_target(build_features(prices))
        test = frame[frame.timestamp.isin(pd.to_datetime(nifty.timestamp))]
        report["comparable_legacy"] = []
        for name in ["logistic_regression", "random_forest", "xgboost"]:
            run = db.scalar(select(ModelRun).where(ModelRun.model_name == name,
                ModelRun.feature_set.in_(["technical_v1", "technical_v1:NIFTY50"])).order_by(desc(ModelRun.created_at)))
            if run:
                bundle = joblib.load(run.artifact_path)
                assert pd.Timestamp(bundle["trained_until"]) < test.timestamp.min()
                result = metrics(test.target_up.astype(int), bundle["pipeline"].predict_proba(test[bundle["features"]])[:, 1])
                report["comparable_legacy"].append({"model": name, "metrics": result, "artifact": Path(run.artifact_path).name,
                                                    "note": "Frozen legacy model evaluated on the exact same new holdout sessions; legacy training ended earlier."})
        enriched, columns = expanded_features(prices)
        correlation = enriched[columns].corr().abs()
        pairs = [{"a": a, "b": b, "correlation": float(correlation.loc[a, b])}
                 for i, a in enumerate(columns) for b in columns[i+1:] if correlation.loc[a, b] > .95]
        report["diagnostics"] = {
            "missing_fraction": enriched[columns].isna().mean().to_dict(),
            "highly_correlated_features": pairs,
            "feature_count": len(columns), "legacy_feature_count": len(FEATURE_COLUMNS),
            "target_sensitivity": [{"horizon": h, "threshold": threshold,
                "up_fraction": float((build_target(enriched, h).future_return.dropna() > threshold).mean())}
                for h in [1, 3, 5] for threshold in [0, .0015]],
        }
        # Moving-block interval preserves short local dependence; no tuning uses it.
        correct = ((nifty.probability.to_numpy() >= .5) == nifty.target_up.to_numpy()).astype(float)
        rng = np.random.default_rng(42)
        means = []
        for _ in range(1000):
            starts = rng.integers(0, len(correct)-10+1, size=int(np.ceil(len(correct)/10)))
            sample = np.concatenate([correct[i:i+10] for i in starts])[:len(correct)]
            means.append(sample.mean())
        report["selected"]["accuracy_block_interval_95"] = np.quantile(means, [.025, .975]).tolist()
        # Global SHAP describes the frozen HOLDOUT model, not a refit on the holdout.
        from src.ml.research import context_frames, dataset, final_evaluation
        frames = {s: load_asset_frame(db, s)[1] for s in ["NIFTY50", "BANKNIFTY", "SENSEX"]}
        market, regimes = context_frames(frames)
        data, sets = dataset(prices, market, regimes)
        data = build_target(data).dropna(subset=["target_up"])
        cols = sets[report["selected"]["feature_set"]]
        _, _, model, _, _ = final_evaluation(data, cols, report["selected"])
        import shap
        x = model[:-1].transform(data.iloc[int(len(data)*.85):][cols])
        values = np.asarray(shap.TreeExplainer(model[-1])(x).values)
        if values.ndim == 3:
            values = values[:, :, 1]
        report["feature_importance"] = sorted([{"feature": c, "mean_absolute_shap": float(v)}
            for c, v in zip(cols, np.abs(values).mean(axis=0), strict=True)], key=lambda x: -x["mean_absolute_shap"])
        report["backtests"] = []
        for strategy in ["buy_hold", "sma_cross", "ml", "ml_threshold", "ml_environment_filter", "regime_aware", "hybrid"]:
            try:
                result = execute_backtest(db, "NIFTY50", strategy)
                report["backtests"].append({"strategy": strategy, "metrics": result["metrics"],
                    "assumptions": result["assumptions"], "start": result["equity_curve"][0]["timestamp"],
                    "end": result["equity_curve"][-1]["timestamp"]})
                print(strategy, result["metrics"], flush=True)
            except ValueError as exc:
                report["backtests"].append({"strategy": strategy, "unavailable": str(exc)})
        save_report(report)
        print("Comparable legacy", report["comparable_legacy"], flush=True)


if __name__ == "__main__":
    main()
