from datetime import UTC, datetime

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.database import Base, get_db
from backend.app.main import app
from backend.app.models import Asset, AssetPrice


def test_api_health_assets_and_prices(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False}
    )
    Session = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)
    with Session() as db:
        a = Asset(
            symbol="TEST",
            name="Test Asset",
            asset_type="stock",
            exchange="NSE",
            sector="IT",
            provider_symbol="TEST.NS",
        )
        db.add(a)
        db.flush()
        db.add(
            AssetPrice(
                asset_id=a.id,
                timestamp=datetime(2025, 1, 1, tzinfo=UTC),
                open=100,
                high=103,
                low=99,
                close=102,
                adjusted_close=102,
                volume=1000,
                provider="test",
                interval="1d",
            )
        )
        db.commit()

    def override():
        with Session() as db:
            yield db

    app.dependency_overrides[get_db] = override
    client = TestClient(app)
    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/api/v1/assets").json()[0]["symbol"] == "TEST"
    assert client.get("/api/v1/assets/TEST/prices").json()[0]["close"] == 102
    with client.websocket_connect("/ws/market") as websocket:
        message = websocket.receive_json()
        assert message["type"] == "market_snapshot"
        assert "freshness" in message["payload"]
    app.dependency_overrides.clear()
