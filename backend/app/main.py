import asyncio
import logging
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api import router
from backend.app.config import get_settings
from backend.app.database import SessionLocal
from backend.app.services import overview

settings = get_settings()
logging.basicConfig(
    level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
app = FastAPI(
    title="Bharat Market Intelligence API",
    version="1.0.0",
    description="Academic market analytics; not investment advice.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(router)
frontend = Path(__file__).resolve().parents[2] / "frontend"
app.mount("/static", StaticFiles(directory=frontend), name="frontend")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(frontend / "index.html")


@app.websocket("/ws/market")
async def market_socket(socket: WebSocket):
    await socket.accept()
    try:
        while True:
            with SessionLocal() as db:
                payload = overview(db)
            message = jsonable_encoder(
                {"type": "market_snapshot", "payload": payload},
                custom_encoder={
                    date: lambda value: value.isoformat(),
                    datetime: lambda value: value.isoformat(),
                    pd.Timestamp: lambda value: value.isoformat(),
                    np.generic: lambda value: value.item(),
                },
            )
            await socket.send_json(message)
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        return
