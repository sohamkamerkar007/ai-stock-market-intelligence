from datetime import UTC, datetime, timedelta

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
        for day in range(90):
            close = 100 + day * 0.2
            db.add(
                AssetPrice(
                    asset_id=a.id,
                    timestamp=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=day),
                    open=close - 0.1,
                    high=close + 1,
                    low=close - 1,
                    close=close,
                    adjusted_close=close,
                    volume=1000 + day,
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
    assert client.get("/api/v1/assets/TEST/prices").json()[0]["close"] == 100
    feature_response = client.get("/api/v1/assets/TEST/features?limit=60")
    assert feature_response.status_code == 200
    assert feature_response.json()[0]["return_1d"] is None
    with client.websocket_connect("/ws/market") as websocket:
        message = websocket.receive_json()
        assert message["type"] == "market_snapshot"
        assert "freshness" in message["payload"]
    app.dependency_overrides.clear()
