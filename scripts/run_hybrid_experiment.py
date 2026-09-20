"""Ablation: technical vs regime-aware vs sentiment-aware vs full contextual model."""

import argparse

import pandas as pd
from sqlalchemy import select

from backend.app.database import SessionLocal
from backend.app.models import ModelRun, NewsArticle, NewsSentiment
from scripts.common import load_asset_frame
from src.features import FEATURE_COLUMNS, build_features, build_target
from src.ml.training import train_classifier
from src.nlp.sentiment import effective_market_date
from src.regimes.discovery import causal_regime_labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="NIFTY50")
    args = p.parse_args()
    with SessionLocal() as db:
        asset, prices = load_asset_frame(db, args.symbol)
        data = build_target(build_features(prices))
        data["regime"] = causal_regime_labels(data)
        news = db.execute(
            select(NewsArticle, NewsSentiment)
            .join(NewsSentiment)
            .order_by(NewsArticle.published_at)
        ).all()
        relevant_news = [
            (article, sentiment)
            for article, sentiment in news
            if asset.asset_type == "index" or args.symbol.upper() in (article.symbols or [])
        ]
        if relevant_news:
            nd = (
                pd.DataFrame(
                    {
                        "timestamp": [
                            effective_market_date(article.published_at)
                            for article, _ in relevant_news
                        ],
                        "sentiment_mean": [sentiment.score for _, sentiment in relevant_news],
                    }
                )
                .groupby("timestamp", as_index=False)
                .mean()
            )
            data = pd.merge_asof(
                data.sort_values("timestamp"),
                nd,
                on="timestamp",
                direction="backward",
                tolerance=pd.Timedelta("3D"),
            )
        sets = {
            "technical_v2": FEATURE_COLUMNS,
            "technical_regime_causal_v2": FEATURE_COLUMNS + ["regime"],
            "technical_regime_sentiment_causal_v2": FEATURE_COLUMNS + ["regime", "sentiment_mean"],
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
