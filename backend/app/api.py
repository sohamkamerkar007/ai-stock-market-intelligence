from datetime import UTC, datetime

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from backend.app.config import get_settings
from backend.app.database import get_db
from backend.app.models import (
    Anomaly,
    Asset,
    MarketRegime,
    ModelRun,
    NewsArticle,
    NewsSentiment,
    Prediction,
    PredictionExplanation,
)
from backend.app.schemas import AssetOut, BacktestRequest, NewsAnalysisRequest, PriceOut
from backend.app.services import execute_backtest, overview, price_frame
from src.features.pipeline import build_features
from src.nlp.sentiment import VaderFinancialBaseline

router = APIRouter(prefix="/api/v1")


@router.get("/health")
def health(db: Session = Depends(get_db)):
    db.execute(select(1))
    return {"status": "ok", "time": datetime.now(UTC).isoformat()}


@router.get("/assets", response_model=list[AssetOut])
def assets(db: Session = Depends(get_db)):
    return db.scalars(
        select(Asset).where(Asset.active.is_(True)).order_by(Asset.asset_type, Asset.symbol)
    ).all()


@router.get("/overview")
def market_overview(db: Session = Depends(get_db)):
    return overview(db)


@router.get("/assets/{symbol}/prices", response_model=list[PriceOut])
def prices(symbol: str, limit: int = Query(365, ge=1, le=5000), db: Session = Depends(get_db)):
    try:
        _, frame = price_frame(db, symbol, limit)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    return frame.to_dict("records")


@router.get("/assets/{symbol}/features")
def features(symbol: str, limit: int = Query(365, ge=60, le=3000), db: Session = Depends(get_db)):
    try:
        _, frame = price_frame(db, symbol, limit)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    if len(frame) < 30:
        raise HTTPException(409, "Insufficient price history; run data ingestion first")
    values = build_features(frame).tail(120).astype(object)
    values = values.where(pd.notna(values), None)
    return values.to_dict("records")


@router.get("/regimes")
def regimes(
    symbol: str = "NIFTY50",
    algorithm: str | None = None,
    limit: int = Query(250, le=2000),
    db: Session = Depends(get_db),
):
    selected_algorithm = algorithm or get_settings().regime_algorithm
    rows = db.execute(
        select(MarketRegime, Asset)
        .join(Asset)
        .where(
            Asset.symbol == symbol.upper(),
            MarketRegime.algorithm == selected_algorithm,
        )
        .order_by(desc(MarketRegime.timestamp))
        .limit(limit)
    ).all()
    return [
        {
            "timestamp": r.timestamp,
            "algorithm": r.algorithm,
            "state": r.state,
            "label": r.label,
            "probability": r.probability,
        }
        for r, _ in rows
    ]


@router.get("/predictions")
def predictions(symbol: str | None = None, db: Session = Depends(get_db)):
    query = (
        select(Prediction, Asset, ModelRun)
        .join(Asset, Prediction.asset_id == Asset.id)
        .outerjoin(ModelRun, Prediction.model_run_id == ModelRun.id)
    )
    if symbol:
        query = query.where(Asset.symbol == symbol.upper())
    rows = db.execute(query.order_by(desc(Prediction.generated_at)).limit(100)).all()
    return [
        {
            "id": p.id,
            "symbol": a.symbol,
            "prediction_for": p.prediction_for,
            "direction": p.direction,
            "probability_up": p.probability_up,
            "expected_return": p.expected_return,
            "regime": p.regime,
            "model": m.model_name if m else None,
            "generated_at": p.generated_at,
        }
        for p, a, m in rows
    ]


@router.get("/predictions/{prediction_id}/explanation")
def explanation(prediction_id: int, db: Session = Depends(get_db)):
    item = db.scalar(
        select(PredictionExplanation).where(PredictionExplanation.prediction_id == prediction_id)
    )
    if not item:
        raise HTTPException(404, "Explanation is unavailable for this prediction")
    return {"summary": item.summary, "contributions": item.contributions}


@router.get("/news")
def news(symbol: str | None = None, limit: int = Query(50, le=200), db: Session = Depends(get_db)):
    fetch_limit = 1000 if symbol else limit
    rows = db.execute(
        select(NewsArticle, NewsSentiment)
        .outerjoin(NewsSentiment)
        .order_by(desc(NewsArticle.published_at))
        .limit(fetch_limit)
    ).all()
    result = [
        {
            "title": a.title,
            "source": a.source,
            "url": a.url,
            "published_at": a.published_at,
            "symbols": a.symbols,
            "sentiment": s.label if s else None,
            "score": s.score if s else None,
        }
        for a, s in rows
    ]
    return [r for r in result if not symbol or symbol.upper() in r["symbols"]][:limit]


@router.post("/news/analyze")
def analyze_news(request: NewsAnalysisRequest, db: Session = Depends(get_db)):
    """Analyze user-provided text without persisting it or presenting it as a forecast."""
    symbol = request.symbol.upper() if request.symbol else None
    asset = db.scalar(select(Asset).where(Asset.symbol == symbol)) if symbol else None
    if symbol and not asset:
        raise HTTPException(404, "The selected asset is not in the configured market universe")
    result = VaderFinancialBaseline().analyze(request.text)
    sentences = [part.strip() for part in request.text.replace("\n", " ").split(".") if part.strip()]
    summary = ". ".join(sentences[:2])[:420]
    if summary and not summary.endswith("."):
        summary += "."
    latest_prediction = None
    if asset:
        latest_prediction = db.scalar(
            select(Prediction)
            .where(Prediction.asset_id == asset.id)
            .order_by(desc(Prediction.generated_at))
        )
    market = overview(db).get("regime")
    agreement = "No asset was selected, so the news tone was not compared with an asset prediction."
    if latest_prediction:
        aligned = (result.score > 0 and latest_prediction.direction == "UP") or (
            result.score < 0 and latest_prediction.direction == "DOWN"
        )
        if result.label == "neutral":
            agreement = "The news tone is neutral and does not strongly support either model direction."
        elif aligned:
            agreement = f"The news tone points in the same direction as the latest {symbol} model output. This agreement does not establish causation."
        else:
            agreement = f"The news tone differs from the latest {symbol} model output, indicating mixed signals."
    return {
        "summary": summary or request.text[:420],
        "sentiment": result.label,
        "score": result.score,
        "confidence": result.confidence,
        "model": result.model,
        "asset": symbol,
        "market_environment": market.get("label") if market else None,
        "signal_comparison": agreement,
        "caveat": "Text sentiment is contextual evidence, not a prediction of future price movement.",
    }


@router.get("/anomalies")
def anomalies(symbol: str | None = None, db: Session = Depends(get_db)):
    query = select(Anomaly, Asset).join(Asset)
    if symbol:
        query = query.where(Asset.symbol == symbol.upper())
    rows = db.execute(query.order_by(desc(Anomaly.timestamp)).limit(200)).all()
    return [
        {
            "symbol": a.symbol,
            "timestamp": x.timestamp,
            "detector": x.detector,
            "type": x.anomaly_type,
            "score": x.score,
            "severity": x.severity,
            "context": x.context,
        }
        for x, a in rows
    ]


@router.get("/research/models")
def model_runs(db: Session = Depends(get_db)):
    rows = db.scalars(select(ModelRun).order_by(desc(ModelRun.created_at)).limit(100)).all()
    return [
        {
            "run_id": row.run_id,
            "task": row.task,
            "model_name": row.model_name,
            "feature_set": row.feature_set,
            "trained_from": row.trained_from,
            "trained_to": row.trained_to,
            "metrics": row.metrics,
            "parameters": row.parameters,
            "created_at": row.created_at,
        }
        for row in rows
    ]


@router.post("/backtests")
def backtest(request: BacktestRequest, db: Session = Depends(get_db)):
    try:
        return execute_backtest(db, request.symbol, request.strategy, request.start, request.end)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
