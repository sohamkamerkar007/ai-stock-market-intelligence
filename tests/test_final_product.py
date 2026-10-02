from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from backend.app.database import Base, get_db
from backend.app.main import app
from backend.app.models import AssetPrice
from src.data.ingestion import ingest_history, seed_assets
from src.data.providers import freshness_status
from src.ml.final_prediction import development_split, load_research_panel
from src.ml.real_direction import infer as infer_real_direction
from src.ml.real_direction import prepare_panel, split_panel
from src.ml.real_direction import train as train_real_direction
from src.ml.real_stock_forecast import prepare, split
from src.ml.stock_clustering import cluster_stocks


def _prices(seed=5, days=500, drift=0.0002):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, days)))
    timestamps = pd.bdate_range("2020-01-01", periods=days)
    return pd.DataFrame({"timestamp": timestamps, "open": close * 0.999,
                         "high": close * 1.01, "low": close * 0.99, "close": close,
                         "volume": rng.integers(100_000, 500_000, days)})


def test_real_targets_are_future_derived_and_boundaries_purged():
    prices = _prices()
    prepared = prepare(prices, 3)
    assert np.isclose(prepared.future_return.iloc[100], prices.close.iloc[103] / prices.close.iloc[100] - 1)
    assert pd.isna(prepared.future_return.iloc[-1])
    training, validation, test = split(prepared)
    assert training.label_available_at.max() < validation.timestamp.min()
    assert validation.label_available_at.max() < test.timestamp.min()
    altered = prices.copy()
    altered.loc[200:, "close"] *= 5
    assert np.isclose(prepare(prices, 3).loc[100, "sma_ratio_20"], prepare(altered, 3).loc[100, "sma_ratio_20"])


def test_real_direction_panel_uses_past_features_and_purged_dates():
    observations = {f"S{i}": _prices(i, 600) for i in range(4)}
    panel = prepare_panel(observations, 3)
    training, validation, test = split_panel(panel)
    assert training.label_available_at.max() < validation.timestamp.min()
    assert validation.label_available_at.max() < test.timestamp.min()
    first = panel[(panel.symbol == "S0")].reset_index(drop=True)
    assert np.isclose(first.future_return.iloc[100],
                      observations["S0"].close.iloc[103] / observations["S0"].close.iloc[100] - 1)


def test_real_direction_inference_uses_persisted_feature_schema():
    observations = {f"S{i}": _prices(i, 600) for i in range(4)}
    artifact = train_real_direction(observations, 1)
    result = infer_real_direction(artifact, observations["S0"])
    assert artifact["features"]
    assert artifact["selected"] in artifact["models"]
    assert 0 <= result["probability_up"] <= 1
    assert result["as_of"] == str(observations["S0"].timestamp.iloc[-1].date())
    assert result["reasons"]


def test_cluster_is_descriptive_and_uses_real_input():
    observations = {f"S{i}": _prices(i, drift=(i - 3) * 0.0003) for i in range(6)}
    result = cluster_stocks(observations, {symbol: "Test" for symbol in observations})
    assert result["status"] == "ok"
    assert len(result["stocks"]) == 6
    assert all(info["cluster"]["count"] >= 1 for info in result["stocks"].values())


def test_cluster_never_crosses_sector_boundaries():
    sectors = {"RELIANCE": "Energy", "ONGC": "Energy", "NTPC": "Energy",
               "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking",
               "TCS": "IT", "INFY": "IT", "WIPRO": "IT"}
    observations = {symbol: _prices(i) for i, symbol in enumerate(sectors)}
    result = cluster_stocks(observations, sectors)
    assert result["status"] == "ok"
    for symbol, sector in sectors.items():
        stock = result["stocks"][symbol]
        assert stock["sector"] == sector
        assert all(peer["sector"] == sector for peer in stock["similar_stocks"])


def test_old_provider_observation_is_stale_even_when_exchange_closed():
    assert freshness_status(datetime.now(UTC) - timedelta(days=10))["status"] == "STALE"


def test_controlled_dataset_chronology_and_class_balance():
    try:
        frame = load_research_panel()
    except FileNotFoundError:
        return
    training, validation, test = development_split(frame, 3)
    assert training.target_available_at_3d.max() < validation.timestamp.min()
    assert validation.target_available_at_3d.max() < test.timestamp.min()
    assert 0.35 < test.target_3d.mean() < 0.65


def test_four_page_api_endpoints_with_empty_database(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'empty.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)

    def override():
        with session() as db:
            yield db

    app.dependency_overrides[get_db] = override
    try:
        client = TestClient(app)
        assert client.get("/api/v1/market/stocks").json() == []
        assert client.get("/api/v1/predictions/final/options").status_code == 200
        assert client.post("/api/v1/predictions/final", json={"symbol": "INVALID", "horizon": 3}).status_code == 404
        assert client.post("/api/v1/predictions/final", json={"symbol": "RELIANCE", "horizon": 2}).status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_incremental_refresh_updates_latest_bar_and_keeps_history_on_failure(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'refresh.db'}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    seed_assets(session)

    class Provider:
        name = "test_provider"
        close = 100.0
        fail = False

        def history(self, symbol, start, end):
            if self.fail:
                raise RuntimeError("provider unavailable")
            return pd.DataFrame({"timestamp": [pd.Timestamp(datetime.now(UTC).date(), tz="UTC")],
                                 "open": [99.0], "high": [102.0], "low": [98.0],
                                 "close": [self.close], "volume": [1000.0]})

    provider = Provider()
    assert ingest_history(session, symbols=["RELIANCE"], provider=provider)["written"] == 1
    provider.close = 101.0
    assert ingest_history(session, symbols=["RELIANCE"], provider=provider)["written"] == 0
    assert session.scalar(select(func.count()).select_from(AssetPrice)) == 1
    assert session.scalar(select(AssetPrice.close)) == 101.0
    provider.fail = True
    assert ingest_history(session, symbols=["RELIANCE"], provider=provider)["status"] == "partial"
    assert session.scalar(select(AssetPrice.close)) == 101.0
    session.close()
