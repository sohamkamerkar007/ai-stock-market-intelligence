"""Read-only presentation of measured experiments and categorical regimes."""

import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.config import get_settings
from backend.app.database import get_db
from backend.app.models import MarketRegime
from backend.app.services import price_frame
from src.features.pipeline import build_features

router = APIRouter(prefix="/api/v1")


@router.get("/research/experiments")
def experiments():
    path = Path(__file__).resolve().parents[2] / "docs" / "experiment_results.json"
    if not path.exists():
        raise HTTPException(
            409, "Run python -m scripts.run_research_matrix to produce measured results"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def profile(frame, state):
    average_return = float(frame.return_1d.mean())
    volatility = float(frame.volatility_20.mean())
    momentum = float(frame.momentum_10.mean())
    drawdown = float(frame.drawdown_60.mean())
    direction = (
        "Positive returns"
        if average_return > 0.0003
        else "Weak returns"
        if average_return < -0.0003
        else "Range-bound"
    )
    risk = (
        "high volatility"
        if volatility > 0.20
        else "moderate volatility"
        if volatility > 0.12
        else "lower volatility"
    )
    trend = "positive momentum" if momentum > 0 else "weak momentum"
    return {
        "state": int(state),
        "name": f"{direction} · {risk} · {trend}",
        "average_return": average_return,
        "volatility": volatility,
        "momentum": momentum,
        "drawdown": drawdown,
        "observations": len(frame),
        "start": frame.timestamp.min().isoformat(),
        "end": frame.timestamp.max().isoformat(),
    }


@router.get("/regimes/summary")
def regime_summary(symbol: str = "NIFTY50", db: Session = Depends(get_db)):
    try:
        asset, prices = price_frame(db, symbol, 10000)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    rows = db.scalars(
        select(MarketRegime)
        .where(
            MarketRegime.asset_id == asset.id,
            MarketRegime.algorithm == get_settings().regime_algorithm,
        )
        .order_by(MarketRegime.timestamp)
    ).all()
    if not rows or prices.empty:
        return {"current": None, "environments": [], "periods": [], "transitions": []}
    features = build_features(prices)
    states = pd.DataFrame(
        [{"timestamp": r.timestamp, "state": r.state} for r in rows]
    ).drop_duplicates("timestamp")
    frame = features.merge(states, on="timestamp", how="inner", validate="one_to_one")
    frame = frame.dropna(subset=["return_1d", "volatility_20", "momentum_10", "drawdown_60"])
    frame["period"] = frame.state.ne(frame.state.shift()).cumsum()
    environments = [profile(g, state) for state, g in frame.groupby("state")]
    names = {p["state"]: p["name"] for p in environments}
    periods = []
    for _, group in frame.groupby("period"):
        item = profile(group, group.state.iloc[0])
        item["name"] = names[item["state"]]
        periods.append(item)
    transitions = [
        {
            "date": b["start"],
            "from": a["name"],
            "to": b["name"],
            "return_change": b["average_return"] - a["average_return"],
            "volatility_change": b["volatility"] - a["volatility"],
            "momentum_change": b["momentum"] - a["momentum"],
            "drawdown_change": b["drawdown"] - a["drawdown"],
        }
        for a, b in zip(periods[:-1], periods[1:], strict=True)
    ]
    return {
        "current": periods[-1] if periods else None,
        "environments": environments,
        "periods": periods,
        "transitions": transitions[-12:][::-1],
        "interpretation": "Retrospective clustering fitted to the available history; these descriptive labels are not used as historical ML features. Predictive experiments use separately fitted causal GMM states.",
        "thresholds": "Mean daily return ±0.03%; annualised volatility low <12%, moderate 12–20%, high >20%; momentum sign. These are descriptive conventions, not forecasts.",
    }
