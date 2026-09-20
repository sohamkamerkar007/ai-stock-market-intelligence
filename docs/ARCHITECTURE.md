# Architecture

The monorepo is a modular research system, not microservices. Provider adapters normalize external observations. SQLAlchemy models and Alembic migrations make PostgreSQL the durable source. Analytical modules consume DataFrames loaded from that store and persist only observed/calculated outputs. FastAPI exposes those records to the static SPA.

```mermaid
flowchart LR
  P[Market/news providers] --> I[Adapters + validation + retries]
  I --> D[(PostgreSQL)]
  D --> F[Feature pipeline]
  F --> S[Supervised models]
  F --> R[Regime discovery]
  F --> A[Anomaly ensemble]
  R --> H[Hybrid ablations]
  S --> X[SHAP translator]
  H --> B[Walk-forward backtests]
  D --> API[FastAPI REST + WebSocket]
  S --> D
  R --> D
  A --> D
  B --> D
  API --> UI[Vanilla JS terminal UI]
```

Key boundaries:

- `src/data`: provider abstraction, validation, universe and idempotent ingestion.
- `src/features`: the sole feature definition; callers do not reimplement indicators.
- `src/ml`, `src/regimes`, `src/nlp`, `src/anomaly`, `src/explainability`, `src/backtesting`: research components.
- `backend/app`: configuration, database models, services, REST/WebSocket transport.
- `frontend`: semantic HTML, CSS design system, modular fetch/WebSocket/chart code.
- `scripts`: reproducible operations and experiment entry points.

The WebSocket broadcasts persisted snapshots every 30 seconds. It does not manufacture ticks. A provider-specific ingestion scheduler can update PostgreSQL independently without changing API/UI contracts.

