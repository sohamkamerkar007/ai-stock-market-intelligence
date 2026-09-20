from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class AssetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    symbol: str
    name: str
    asset_type: str
    exchange: str
    sector: str | None


class PriceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class BacktestRequest(BaseModel):
    symbol: str
    strategy: str = Field(pattern="^(buy_hold|sma_cross|ml|regime_aware|hybrid)$")
    start: date | None = None
    end: date | None = None
