"""Four-page product APIs; research and real-market model domains stay separate."""

import json
from datetime import UTC, datetime

import joblib
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models import Asset, AssetPrice
from backend.app.services import price_frame
from src.data.ingestion import ingest_history, seed_assets
from src.data.universe import UNIVERSE
from src.ml.final_prediction import ROOT
from src.ml.real_direction import infer as infer_direction
from src.ml.real_stock_forecast import infer
from src.ml.stock_clustering import cluster_stocks

router = APIRouter(prefix="/api/v1")
STOCK_SECTORS = {item["symbol"]: item["sector"] for item in UNIVERSE if item["type"] == "stock"}
ALLOWED_SECTORS = frozenset(STOCK_SECTORS.values())


class FinalPredictionRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=32)
    horizon: int = Field(default=3, ge=1, le=5)


class RefreshRequest(BaseModel):
    symbol: str | None = None


@router.post("/market/refresh")
def refresh_market(request: RefreshRequest, db: Session = Depends(get_db)):
    seed_assets(db)
    symbol = request.symbol.upper() if request.symbol else None
    if symbol and symbol not in {item["symbol"] for item in UNIVERSE}:
        raise HTTPException(404, "Unknown tracked asset")
    result = ingest_history(db, symbols=[symbol] if symbol else None)
    latest = db.scalar(select(AssetPrice.timestamp).order_by(AssetPrice.timestamp.desc()).limit(1))
    return {**result, "as_of": latest, "refreshed_at": datetime.now(UTC),
            "message": ("Latest available market data fetched." if not result["failures"] else
                        "Unable to refresh some market data right now. Showing the latest available data.")}


@router.get("/market/stocks")
def stock_screener(sector: str | None = Query(None), db: Session = Depends(get_db)):
    if sector and sector not in ALLOWED_SECTORS:
        raise HTTPException(422, "Unknown sector")
    result = []
    for asset in db.scalars(select(Asset).where(Asset.asset_type == "stock", Asset.active.is_(True)).order_by(Asset.symbol)):
        authoritative_sector = STOCK_SECTORS.get(asset.symbol)
        if sector and authoritative_sector != sector:
            continue
        _, prices = price_frame(db, asset.symbol, 2)
        if prices.empty:
            continue
        last = prices.iloc[-1]
        previous = prices.iloc[-2] if len(prices) > 1 else None
        result.append({"symbol": asset.symbol, "name": asset.name, "sector": authoritative_sector,
                       "close": float(last.close), "change": float(last.close / previous.close - 1) if previous is not None else None,
                       "change_amount": float(last.close - previous.close) if previous is not None else None,
                       "volume": float(last.volume), "as_of": last.timestamp})
    return result


@router.get("/assets/{symbol}/cluster")
def stock_cluster(symbol: str, db: Session = Depends(get_db)):
    stocks = db.scalars(select(Asset).where(Asset.asset_type == "stock", Asset.active.is_(True))).all()
    if symbol.upper() not in {stock.symbol for stock in stocks}:
        raise HTTPException(404, "Unknown tracked stock")
    selected_sector = STOCK_SECTORS.get(symbol.upper())
    if not selected_sector:
        return {"status": "insufficient_data", "message": "Sector unavailable for this stock."}
    stocks = [stock for stock in stocks if STOCK_SECTORS.get(stock.symbol) == selected_sector]
    observations = {stock.symbol: price_frame(db, stock.symbol, 252)[1] for stock in stocks}
    result = cluster_stocks(observations, {stock.symbol: STOCK_SECTORS.get(stock.symbol) for stock in stocks})
    if result["status"] != "ok":
        return result
    return {"status": "ok", "method": result["method"], "stock": result["stocks"][symbol.upper()],
            "groups": result["groups"], "coverage": len(result["stocks"])}


@router.get("/predictions/final/options")
def final_options(db: Session = Depends(get_db)):
    assets = db.scalars(select(Asset).where(Asset.asset_type == "stock", Asset.active.is_(True)).order_by(Asset.symbol)).all()
    report_path = ROOT / "docs/final_classification_results.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    return {"stocks": [{"symbol": a.symbol, "name": a.name, "sector": a.sector} for a in assets],
            "horizons": [1, 3, 5],
            "research": report}


@router.post("/predictions/final")
def final_prediction(request: FinalPredictionRequest, db: Session = Depends(get_db)):
    if request.horizon not in (1, 3, 5):
        raise HTTPException(422, "Choose 1, 3, or 5 trading sessions")
    symbol = request.symbol.upper()
    try:
        _, prices = price_frame(db, symbol, 5000)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    path = ROOT / "models/final" / f"stock_{symbol}_{request.horizon}.joblib"
    if not path.exists():
        raise HTTPException(409, "Real-stock regression artifact unavailable. Run python -m scripts.train_final_stock_models")
    real_result = infer(joblib.load(path), prices)
    direction_path = ROOT / "models/final" / f"real_direction_{request.horizon}.joblib"
    direction_result = None
    if direction_path.exists():
        try:
            direction_result = infer_direction(joblib.load(direction_path), prices)
        except ValueError:
            pass
    return {"symbol": symbol, "data_type": "real_indian_market", "real_stock": real_result,
            "direction": direction_result,
            "controlled_research": None,
            "notice": ("Real-stock direction uses a separately validated classifier on real market history. Its observed test accuracy is not a guarantee for future movement. Synthetic research accuracy does not apply to this stock."
                       if direction_result else "Direction and confidence are unavailable for this real stock. Train the real direction pipeline first; synthetic research accuracy does not apply to this stock.")}
