import logging
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from backend.app.models import Asset, AssetPrice, DataIngestionRun
from src.data.providers import MarketDataProvider, YahooResearchProvider
from src.data.universe import UNIVERSE

logger = logging.getLogger(__name__)


def seed_assets(session: Session) -> int:
    created = 0
    for item in UNIVERSE:
        existing = session.scalar(select(Asset).where(Asset.symbol == item["symbol"]))
        if existing:
            continue
        session.add(
            Asset(
                symbol=item["symbol"],
                name=item["name"],
                asset_type=item["type"],
                exchange=item["exchange"],
                sector=item["sector"],
                provider_symbol=item["provider_symbol"],
            )
        )
        created += 1
    session.commit()
    return created


def ingest_history(
    session: Session,
    years: int = 8,
    symbols: list[str] | None = None,
    provider: MarketDataProvider | None = None,
) -> dict:
    provider = provider or YahooResearchProvider()
    run = DataIngestionRun(provider=provider.name, status="running")
    session.add(run)
    session.commit()
    received = written = 0
    failures: dict[str, str] = {}
    assets = session.scalars(select(Asset).where(Asset.active.is_(True))).all()
    if symbols:
        wanted = {s.upper() for s in symbols}
        assets = [a for a in assets if a.symbol in wanted]
    for asset in assets:
        try:
            latest = session.scalar(
                select(func.max(AssetPrice.timestamp)).where(AssetPrice.asset_id == asset.id)
            )
            start = (
                (latest.date() + timedelta(days=1))
                if latest
                else (date.today() - timedelta(days=365 * years + 30))
            )
            if start >= date.today():
                continue
            frame = provider.history(asset.provider_symbol, start, date.today() + timedelta(days=1))
            received += len(frame)
            for row in frame.to_dict("records"):
                values = dict(
                    asset_id=asset.id,
                    timestamp=row["timestamp"].to_pydatetime(),
                    interval="1d",
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    adjusted_close=float(row.get("adjusted_close", row["close"])),
                    volume=float(row["volume"]),
                    provider=provider.name,
                )
                if session.bind.dialect.name == "postgresql":
                    stmt = (
                        pg_insert(AssetPrice)
                        .values(**values)
                        .on_conflict_do_nothing(constraint="uq_asset_price_bar")
                    )
                    written += session.execute(stmt).rowcount
                elif not session.scalar(
                    select(AssetPrice.id).where(
                        AssetPrice.asset_id == asset.id,
                        AssetPrice.timestamp == values["timestamp"],
                        AssetPrice.interval == "1d",
                    )
                ):
                    session.add(AssetPrice(**values))
                    written += 1
            session.commit()
        except Exception as exc:
            session.rollback()
            failures[asset.symbol] = str(exc)
            logger.exception("Ingestion failed for %s", asset.symbol)
    run = session.get(DataIngestionRun, run.id)
    run.completed_at = datetime.now(UTC)
    run.status = "partial" if failures else "success"
    run.rows_received = received
    run.rows_written = written
    run.details = {"failures": failures}
    session.commit()
    return {"status": run.status, "received": received, "written": written, "failures": failures}
