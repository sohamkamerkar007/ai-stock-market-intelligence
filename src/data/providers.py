import logging
from abc import ABC, abstractmethod
from datetime import UTC, date, datetime

import pandas as pd
import requests
import yfinance as yf
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class MarketDataError(RuntimeError):
    pass


class MarketDataProvider(ABC):
    name: str

    @abstractmethod
    def history(self, symbol: str, start: date, end: date) -> pd.DataFrame: ...


class YahooResearchProvider(MarketDataProvider):
    """Unofficial research/EOD adapter. Not exchange-grade and never labelled live."""

    name = "yahoo_research"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        frame = yf.download(
            symbol,
            start=start.isoformat(),
            end=end.isoformat(),
            progress=False,
            auto_adjust=False,
            timeout=30,
        )
        if frame.empty:
            raise MarketDataError(f"No history returned for {symbol}")
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame = frame.rename(columns=str.lower).reset_index()
        frame.columns = [str(c).lower().replace(" ", "_") for c in frame.columns]
        if "date" in frame:
            frame = frame.rename(columns={"date": "timestamp"})
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["adjusted_close"] = frame.get("adj_close", frame["close"])
        return validate_bars(frame)


class TwelveDataQuoteProvider:
    name = "twelve_data"
    endpoint = "https://api.twelvedata.com/quote"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=15), reraise=True)
    def quote(self, symbol: str) -> dict:
        response = requests.get(
            self.endpoint, params={"symbol": symbol, "apikey": self.api_key}, timeout=15
        )
        if response.status_code == 429:
            raise MarketDataError("Quote provider rate limit reached")
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") == "error":
            raise MarketDataError(payload.get("message", "Quote provider error"))
        return payload


def validate_bars(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise MarketDataError(f"Missing OHLCV columns: {sorted(missing)}")
    clean = frame.copy().drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    numeric = ["open", "high", "low", "close", "volume"]
    clean[numeric] = clean[numeric].apply(pd.to_numeric, errors="coerce")
    clean = clean.dropna(subset=["timestamp", "open", "high", "low", "close"])
    invalid = (clean[["open", "high", "low", "close"]] <= 0).any(axis=1) | (clean["volume"] < 0)
    invalid |= clean["high"] < clean[["open", "close", "low"]].max(axis=1)
    invalid |= clean["low"] > clean[["open", "close", "high"]].min(axis=1)
    if invalid.any():
        logger.warning("Discarding %s invalid OHLCV rows", int(invalid.sum()))
        clean = clean.loc[~invalid]
    if clean.empty:
        raise MarketDataError("All returned market bars failed validation")
    return clean


def freshness_status(last_timestamp: datetime | None, delay_minutes: int = 15) -> dict:
    if last_timestamp is None:
        return {"status": "UNAVAILABLE", "as_of": None, "age_minutes": None}
    now = datetime.now(UTC)
    stamp = last_timestamp if last_timestamp.tzinfo else last_timestamp.replace(tzinfo=UTC)
    age = max(0, (now - stamp).total_seconds() / 60)
    if now.weekday() >= 5 or not (3.75 <= now.hour + now.minute / 60 <= 10):
        status = "MARKET_CLOSED"
    elif age <= max(2, delay_minutes):
        status = "DELAYED" if delay_minutes else "LIVE"
    else:
        status = "STALE"
    return {"status": status, "as_of": stamp.isoformat(), "age_minutes": round(age, 1)}
