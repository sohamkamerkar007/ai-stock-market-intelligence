from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

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
from backend.app.schemas import AssetOut, BacktestRequest, PriceOut
from backend.app.services import execute_backtest, overview, price_frame
from src.features.pipeline import build_features

router = APIRouter(prefix="/api/v1")


@router.get("/health")
def health(db: Session = Depends(get_db)):
    db.execute(select(1))
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}


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
    values = build_features(frame).tail(120).where(lambda x: x.notna(), None)
    return values.to_dict("records")


@router.get("/regimes")
def regimes(
    symbol: str = "NIFTY50", limit: int = Query(250, le=2000), db: Session = Depends(get_db)
):
    rows = db.execute(
        select(MarketRegime, Asset)
        .join(Asset)
        .where(Asset.symbol == symbol.upper())
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
    rows = db.execute(
        select(NewsArticle, NewsSentiment)
        .outerjoin(NewsSentiment)
        .order_by(desc(NewsArticle.published_at))
        .limit(limit)
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
    return [r for r in result if not symbol or symbol.upper() in r["symbols"]]


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
    return db.scalars(select(ModelRun).order_by(desc(ModelRun.created_at)).limit(100)).all()


@router.post("/backtests")
def backtest(request: BacktestRequest, db: Session = Depends(get_db)):
    try:
        return execute_backtest(db, request.symbol, request.strategy, request.start, request.end)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
