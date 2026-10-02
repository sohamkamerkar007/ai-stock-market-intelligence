"""Train saved real-stock direction models on a purged chronological split."""

import json
from datetime import UTC, datetime

import joblib
from sqlalchemy import select

from backend.app.database import SessionLocal
from backend.app.models import Asset
from backend.app.services import price_frame
from src.ml.final_prediction import ROOT
from src.ml.real_direction import train


def main() -> None:
    with SessionLocal() as db:
        assets = db.scalars(select(Asset).where(Asset.asset_type == "stock", Asset.active.is_(True))).all()
        observations = {asset.symbol: price_frame(db, asset.symbol, 5000)[1] for asset in assets}
    output = ROOT / "models/final"
    output.mkdir(parents=True, exist_ok=True)
    report = {"created_at": datetime.now(UTC).isoformat(), "data_type": "real_indian_market",
              "selection": "Highest validation F1, then validation accuracy; test set untouched", "horizons": {}}
    for horizon in (1, 3, 5):
        artifact = train(observations, horizon)
        joblib.dump(artifact, output / f"real_direction_{horizon}.joblib")
        report["horizons"][str(horizon)] = {
            key: value for key, value in artifact.items() if key != "models"
        } | {"models": {name: {"validation": value["validation"], "test": value["test"]}
                         for name, value in artifact["models"].items()}}
        print("Trained real direction horizon:", horizon, flush=True)
    (output / "real_direction_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
