import argparse

from sqlalchemy import delete

from backend.app.database import SessionLocal
from backend.app.models import Anomaly
from scripts.common import load_asset_frame
from src.anomaly.detection import detect_anomalies
from src.features import build_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["NIFTY50", "RELIANCE", "TCS", "HDFCBANK"])
    args = p.parse_args()
    with SessionLocal() as db:
        for symbol in args.symbols:
            asset, prices = load_asset_frame(db, symbol)
            found = detect_anomalies(build_features(prices))
            db.execute(delete(Anomaly).where(Anomaly.asset_id == asset.id))
            for r in found[found.is_anomaly].itertuples():
                db.add(
                    Anomaly(
                        asset_id=asset.id,
                        timestamp=r.timestamp,
                        detector="isolation_forest+lof",
                        anomaly_type=r.anomaly_type,
                        score=r.anomaly_score,
                        severity=r.severity,
                        context={
                            "return_1d": r.return_1d,
                            "relative_volume_20": r.relative_volume_20,
                            "volatility_20": r.volatility_20,
                        },
                    )
                )
            db.commit()
            print(symbol, int(found.is_anomaly.sum()))


if __name__ == "__main__":
    main()
