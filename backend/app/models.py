from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base


class Asset(Base):
    __tablename__ = "assets"
    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(160))
    asset_type: Mapped[str] = mapped_column(String(20), index=True)
    exchange: Mapped[str] = mapped_column(String(20))
    sector: Mapped[str | None] = mapped_column(String(80), index=True)
    provider_symbol: Mapped[str] = mapped_column(String(40), unique=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    prices: Mapped[list["AssetPrice"]] = relationship(back_populates="asset")


class AssetPrice(Base):
    __tablename__ = "asset_prices"
    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    interval: Mapped[str] = mapped_column(String(12), default="1d")
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    adjusted_close: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float, default=0)
    provider: Mapped[str] = mapped_column(String(40))
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    asset: Mapped[Asset] = relationship(back_populates="prices")
    __table_args__ = (
        UniqueConstraint("asset_id", "timestamp", "interval", name="uq_asset_price_bar"),
        Index("ix_asset_prices_asset_time", "asset_id", "timestamp"),
    )


class TechnicalFeature(Base):
    __tablename__ = "technical_features"
    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    feature_set: Mapped[str] = mapped_column(String(40), default="core_v1")
    values: Mapped[dict[str, Any]] = mapped_column(JSON)
    __table_args__ = (
        UniqueConstraint("asset_id", "timestamp", "feature_set", name="uq_feature_row"),
    )


class MarketRegime(Base):
    __tablename__ = "market_regimes"
    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    algorithm: Mapped[str] = mapped_column(String(32))
    state: Mapped[int] = mapped_column(Integer)
    label: Mapped[str] = mapped_column(String(120))
    probability: Mapped[float | None] = mapped_column(Float)


class ModelRun(Base):
    __tablename__ = "model_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    task: Mapped[str] = mapped_column(String(40))
    model_name: Mapped[str] = mapped_column(String(60))
    feature_set: Mapped[str] = mapped_column(String(40))
    trained_from: Mapped[date | None] = mapped_column(Date)
    trained_to: Mapped[date | None] = mapped_column(Date)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON)
    artifact_path: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"))
    model_run_id: Mapped[int | None] = mapped_column(ForeignKey("model_runs.id"))
    prediction_for: Mapped[date] = mapped_column(Date, index=True)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    direction: Mapped[str] = mapped_column(String(8))
    probability_up: Mapped[float] = mapped_column(Float)
    expected_return: Mapped[float | None] = mapped_column(Float)
    regime: Mapped[str | None] = mapped_column(String(120))


class PredictionExplanation(Base):
    __tablename__ = "prediction_explanations"
    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"), unique=True
    )
    contributions: Mapped[dict[str, Any]] = mapped_column(JSON)
    summary: Mapped[str] = mapped_column(Text)


class NewsArticle(Base):
    __tablename__ = "news_articles"
    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(180), unique=True)
    title: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(100))
    url: Mapped[str] = mapped_column(Text)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    symbols: Mapped[list[str]] = mapped_column(JSON, default=list)
    description: Mapped[str | None] = mapped_column(Text)


class NewsSentiment(Base):
    __tablename__ = "news_sentiment"
    id: Mapped[int] = mapped_column(primary_key=True)
    article_id: Mapped[int] = mapped_column(
        ForeignKey("news_articles.id", ondelete="CASCADE"), unique=True
    )
    model: Mapped[str] = mapped_column(String(40))
    label: Mapped[str] = mapped_column(String(12))
    score: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)


class Anomaly(Base):
    __tablename__ = "anomalies"
    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    detector: Mapped[str] = mapped_column(String(40))
    anomaly_type: Mapped[str] = mapped_column(String(60))
    score: Mapped[float] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String(12))
    context: Mapped[dict[str, Any]] = mapped_column(JSON)


class Backtest(Base):
    __tablename__ = "backtests"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id", ondelete="CASCADE"))
    strategy: Mapped[str] = mapped_column(String(60))
    started_at: Mapped[date] = mapped_column(Date)
    ended_at: Mapped[date] = mapped_column(Date)
    assumptions: Mapped[dict[str, Any]] = mapped_column(JSON)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON)
    equity_curve: Mapped[list[dict[str, Any]]] = mapped_column(JSON)


class DataIngestionRun(Base):
    __tablename__ = "data_ingestion_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(40))
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(20), index=True)
    rows_received: Mapped[int] = mapped_column(Integer, default=0)
    rows_written: Mapped[int] = mapped_column(Integer, default=0)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
