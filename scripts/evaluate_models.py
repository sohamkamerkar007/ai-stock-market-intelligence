from sqlalchemy import select

from backend.app.database import SessionLocal
from backend.app.models import ModelRun

with SessionLocal() as db:
    for run in db.scalars(select(ModelRun).order_by(ModelRun.created_at.desc())).all():
        print(run.run_id, run.model_name, run.feature_set, run.metrics)
