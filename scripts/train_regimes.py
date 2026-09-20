import argparse

import joblib
from sqlalchemy import delete

from backend.app.database import SessionLocal
from backend.app.models import MarketRegime
from scripts.common import load_asset_frame
from src.features import build_features
from src.regimes.discovery import discover_regimes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="NIFTY50")
    p.add_argument("--algorithm", choices=["kmeans", "gmm", "hmm"], default="hmm")
    p.add_argument("--states", type=int, default=3)
    args = p.parse_args()
    with SessionLocal() as db:
        asset, prices = load_asset_frame(db, args.symbol)
        features = build_features(prices)
        result = discover_regimes(features, args.algorithm, args.states)
        db.execute(
            delete(MarketRegime).where(
                MarketRegime.asset_id == asset.id, MarketRegime.algorithm == args.algorithm
            )
        )
        probabilities = (
            result.model.predict_proba(
                result.scaler.transform(
                    features.loc[
                        result.labels.index,
                        [
                            "return_20d",
                            "volatility_20",
                            "momentum_10",
                            "relative_volume_20",
                            "drawdown_60",
                        ],
                    ]
                )
            )
            if hasattr(result.model, "predict_proba")
            else None
        )
        for i, state in result.labels.items():
            db.add(
                MarketRegime(
                    asset_id=asset.id,
                    timestamp=features.loc[i, "timestamp"],
                    algorithm=args.algorithm,
                    state=int(state),
                    label=result.descriptions[int(state)],
                    probability=float(probabilities[list(result.labels.index).index(i), int(state)])
                    if probabilities is not None
                    else None,
                )
            )
        db.commit()
        joblib.dump(
            {"model": result.model, "scaler": result.scaler, "descriptions": result.descriptions},
            f"models/regime-{args.symbol}-{args.algorithm}.joblib",
        )
        print(result.metrics, result.descriptions)


if __name__ == "__main__":
    main()
