import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import Asset, AssetPrice


def load_asset_frame(session: Session, symbol: str) -> tuple[Asset, pd.DataFrame]:
    asset = session.scalar(select(Asset).where(Asset.symbol == symbol.upper()))
    if not asset:
        raise ValueError(f"Unknown asset {symbol}")
    rows = session.scalars(
        select(AssetPrice).where(AssetPrice.asset_id == asset.id).order_by(AssetPrice.timestamp)
    ).all()
    if not rows:
        raise ValueError(f"No prices for {symbol}; run scripts/bootstrap.py")
    return asset, pd.DataFrame(
        [
            {
                "timestamp": x.timestamp,
                "open": x.open,
                "high": x.high,
                "low": x.low,
                "close": x.close,
                "volume": x.volume,
            }
            for x in rows
        ]
    )
