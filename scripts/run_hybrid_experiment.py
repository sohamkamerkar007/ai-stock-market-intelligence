"""Ablation: technical vs regime-aware vs sentiment-aware vs full contextual model."""

import argparse

import pandas as pd
from sqlalchemy import select

from backend.app.database import SessionLocal
from backend.app.models import MarketRegime, ModelRun, NewsArticle, NewsSentiment
from scripts.common import load_asset_frame
from src.features import FEATURE_COLUMNS, build_features, build_target
from src.ml.training import train_classifier


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="NIFTY50")
    args = p.parse_args()
    with SessionLocal() as db:
        asset, prices = load_asset_frame(db, args.symbol)
        data = build_target(build_features(prices))
        regimes = db.scalars(
            select(MarketRegime)
            .where(MarketRegime.asset_id == asset.id)
            .order_by(MarketRegime.timestamp)
        ).all()
        if regimes:
            reg = pd.DataFrame(
                {"timestamp": [r.timestamp for r in regimes], "regime": [r.state for r in regimes]}
            )
            data = pd.merge_asof(
                data.sort_values("timestamp"),
                reg.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )
        news = db.execute(
            select(NewsArticle, NewsSentiment)
            .join(NewsSentiment)
            .order_by(NewsArticle.published_at)
        ).all()
        if news:
            nd = (
                pd.DataFrame(
                    {
                        "timestamp": [a.published_at for a, _ in news],
                        "sentiment_mean": [s.score for _, s in news],
                    }
                )
                .set_index("timestamp")
                .resample("1D")
                .mean()
                .reset_index()
            )
            data = pd.merge_asof(
                data.sort_values("timestamp"),
                nd,
                on="timestamp",
                direction="backward",
                tolerance=pd.Timedelta("3D"),
            )
        sets = {
            "technical": FEATURE_COLUMNS,
            "technical_regime": FEATURE_COLUMNS + ["regime"],
            "technical_regime_sentiment": FEATURE_COLUMNS + ["regime", "sentiment_mean"],
        }
        for label, features in sets.items():
            if any(c not in data or data[c].notna().sum() < 100 for c in features):
                print("SKIP", label, "missing contextual observations")
                continue
            result = train_classifier(
                data.dropna(subset=FEATURE_COLUMNS), features, "logistic_regression"
            )
            db.add(
                ModelRun(
                    run_id=result.run_id,
                    task="hybrid_ablation",
                    model_name="logistic_regression",
                    feature_set=label,
                    trained_from=data.timestamp.min().date(),
                    trained_to=data.timestamp.max().date(),
                    metrics=result.metrics,
                    parameters={"seed": 42, "no_shuffle": True},
                    artifact_path=result.artifact_path,
                )
            )
            db.commit()
            print(label, result.metrics)


if __name__ == "__main__":
    main()
