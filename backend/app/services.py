import pandas as pd
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from backend.app.config import get_settings
from backend.app.models import (
    Anomaly,
    Asset,
    AssetPrice,
    MarketRegime,
    NewsArticle,
    NewsSentiment,
    Prediction,
)
from src.backtesting.engine import run_backtest
from src.data.providers import freshness_status
from src.features.pipeline import FEATURE_COLUMNS, build_features, build_target
from src.ml.training import walk_forward_probabilities
from src.nlp.sentiment import effective_market_date
from src.regimes.discovery import causal_regime_labels


def price_frame(session: Session, symbol: str, limit: int = 1000) -> tuple[Asset, pd.DataFrame]:
    asset = session.scalar(select(Asset).where(Asset.symbol == symbol.upper()))
    if not asset:
        raise LookupError(f"Unknown asset: {symbol}")
    rows = session.scalars(
        select(AssetPrice)
        .where(AssetPrice.asset_id == asset.id)
        .order_by(desc(AssetPrice.timestamp))
        .limit(limit)
    ).all()
    rows = list(reversed(rows))
    return asset, pd.DataFrame(
        [
            {
                "timestamp": r.timestamp,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in rows
        ]
    )


def overview(session: Session) -> dict:
    settings = get_settings()
    indices = []
    for symbol in ["NIFTY50", "BANKNIFTY", "SENSEX"]:
        try:
            asset, frame = price_frame(session, symbol, 2)
            if len(frame):
                change = (
                    (frame.close.iloc[-1] / frame.close.iloc[-2] - 1) if len(frame) > 1 else None
                )
                indices.append(
                    {
                        "symbol": symbol,
                        "name": asset.name,
                        "close": frame.close.iloc[-1],
                        "change": change,
                        "as_of": frame.timestamp.iloc[-1],
                    }
                )
        except LookupError:
            pass
    latest = session.scalar(select(func.max(AssetPrice.timestamp)))
    latest_stock = session.scalar(
        select(func.max(AssetPrice.timestamp))
        .join(Asset)
        .where(Asset.asset_type == "stock", Asset.active.is_(True))
    )
    stock_assets = session.scalars(
        select(Asset).where(Asset.asset_type == "stock", Asset.active.is_(True))
    ).all()
    movers = []
    for asset in stock_assets:
        rows = session.scalars(
            select(AssetPrice)
            .where(AssetPrice.asset_id == asset.id)
            .order_by(desc(AssetPrice.timestamp))
            .limit(2)
        ).all()
        if len(rows) == 2 and latest_stock and rows[0].timestamp.date() == latest_stock.date():
            movers.append(
                {
                    "symbol": asset.symbol,
                    "name": asset.name,
                    "close": rows[0].close,
                    "change": rows[0].close / rows[1].close - 1,
                    "sector": asset.sector,
                }
            )
    up = sum(m["change"] > 0 for m in movers)
    regime = session.execute(
        select(MarketRegime, Asset)
        .join(Asset)
        .where(
            Asset.symbol == "NIFTY50",
            MarketRegime.algorithm == settings.regime_algorithm,
        )
        .order_by(desc(MarketRegime.timestamp))
        .limit(1)
    ).first()
    predictions = session.execute(
        select(Prediction, Asset).join(Asset).order_by(desc(Prediction.generated_at)).limit(5)
    ).all()
    anomalies = session.execute(
        select(Anomaly, Asset).join(Asset).order_by(desc(Anomaly.timestamp)).limit(5)
    ).all()
    return {
        "freshness": freshness_status(latest),
        "indices": indices,
        "breadth": {"advancing": up, "declining": len(movers) - up, "coverage": len(movers)},
        "top_gainers": sorted(movers, key=lambda x: x["change"], reverse=True)[:5],
        "top_losers": sorted(movers, key=lambda x: x["change"])[:5],
        "regime": (
            {
                "label": regime[0].label,
                "probability": regime[0].probability,
                "as_of": regime[0].timestamp,
            }
            if regime
            else None
        ),
        "predictions": [
            {
                "symbol": a.symbol,
                "direction": p.direction,
                "probability_up": p.probability_up,
                "for": p.prediction_for,
            }
            for p, a in predictions
        ],
        "anomalies": [
            {
                "symbol": a.symbol,
                "type": x.anomaly_type,
                "severity": x.severity,
                "timestamp": x.timestamp,
            }
            for x, a in anomalies
        ],
    }


def execute_backtest(session: Session, symbol: str, strategy: str, start=None, end=None) -> dict:
    asset, frame = price_frame(session, symbol, 10000)
    if frame.empty:
        raise ValueError("No prices loaded for this asset")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    if start:
        frame = frame[frame.timestamp.dt.date >= start]
    if end:
        frame = frame[frame.timestamp.dt.date <= end]
    if len(frame) < 60:
        raise ValueError("At least 60 price observations are required")
    features = build_target(build_features(frame))
    if strategy == "buy_hold":
        signal = pd.Series(1.0, index=features.index)
    elif strategy == "sma_cross":
        signal = (features["sma_ratio_10"] > features["sma_ratio_20"]).astype(float)
    else:
        model_features = list(FEATURE_COLUMNS)
        if strategy in {"regime_aware", "hybrid"}:
            features["regime_state"] = causal_regime_labels(features)
            # The causal detector deliberately has no state until its minimum
            # training history exists. Do not let an imputer silently remove an
            # all-null regime column in early walk-forward folds.
            features = features.dropna(subset=["regime_state"])
            if len(features) < 60:
                raise ValueError("Insufficient post-warm-up history for regime backtesting")
            model_features.append("regime_state")
        if strategy == "hybrid":
            news_rows = session.execute(
                select(NewsArticle, NewsSentiment)
                .join(NewsSentiment)
                .order_by(NewsArticle.published_at)
            ).all()
            relevant = [
                {
                    "timestamp": effective_market_date(article.published_at),
                    "sentiment_mean": sentiment.score,
                }
                for article, sentiment in news_rows
                if asset.asset_type == "index" or symbol.upper() in (article.symbols or [])
            ]
            if not relevant:
                raise ValueError(
                    "Hybrid backtest requires time-aligned news for the selected asset"
                )
            sentiment_frame = pd.DataFrame(relevant)
            sentiment_frame["timestamp"] = pd.to_datetime(sentiment_frame["timestamp"], utc=True)
            sentiment_frame = sentiment_frame.groupby("timestamp", as_index=False).mean()
            features = pd.merge_asof(
                features.sort_values("timestamp"),
                sentiment_frame,
                on="timestamp",
                direction="backward",
                tolerance=pd.Timedelta("3D"),
            )
            model_features.append("sentiment_mean")
        probability = walk_forward_probabilities(features, model_features)
        signal = (probability >= 0.55).astype(float)
    result = run_backtest(features, signal)
    return {
        "symbol": asset.symbol,
        "strategy": strategy,
        "metrics": result.metrics,
        "equity_curve": [
            {"timestamp": row.timestamp.isoformat(), "equity": row.equity, "position": row.position}
            for row in result.equity_curve.itertuples()
        ],
    }
