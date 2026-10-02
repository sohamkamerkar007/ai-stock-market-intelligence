"""Persist real-market price/volatility regressors for the tracked stock universe."""

import json
from datetime import UTC, datetime

import joblib
from sqlalchemy import select

from backend.app.database import SessionLocal
from backend.app.models import Asset
from backend.app.services import price_frame
from src.ml.final_prediction import ROOT
from src.ml.real_stock_forecast import train


def main() -> None:
    output = ROOT / "models/final"
    output.mkdir(parents=True, exist_ok=True)
    report = {"created_at": datetime.now(UTC).isoformat(), "dataset_type": "real_indian_market",
              "method": "Chronological 65/15/20 with boundary purging; alpha selected only on validation",
              "assets": {}}
    with SessionLocal() as db:
        symbols = db.scalars(select(Asset.symbol).where(Asset.asset_type == "stock", Asset.active.is_(True))).all()
        for symbol in symbols:
            _, prices = price_frame(db, symbol, 5000)
            report["assets"][symbol] = {}
            for horizon in (1, 3, 5):
                try:
                    artifact = train(prices, horizon)
                except ValueError as exc:
                    report["assets"][symbol][str(horizon)] = {"error": str(exc)}
                    continue
                joblib.dump(artifact, output / f"stock_{symbol}_{horizon}.joblib")
                report["assets"][symbol][str(horizon)] = {
                    "test_start": artifact["test_start"], "test_end": artifact["test_end"],
                    "metrics": {name: value["metrics"] for name, value in artifact["models"].items()},
                    "validation_mae": {name: value["validation_mae"] for name, value in artifact["models"].items()},
                    "selected": {target: min(
                        (name for name in artifact["models"] if name.endswith(target)),
                        key=lambda name: artifact["models"][name]["validation_mae"],
                    ) for target in ("future_return", "future_volatility")},
                }
            print("Trained real-stock regression:", symbol, flush=True)
    (output / "final_stock_regression_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    failed = {symbol: horizon for symbol, horizons in report["assets"].items()
              for horizon, value in horizons.items() if "error" in value}
    if failed:
        raise SystemExit(f"Some regression artifacts could not be trained: {failed}")


if __name__ == "__main__":
    main()
