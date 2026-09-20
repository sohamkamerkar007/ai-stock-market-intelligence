"""Initialize schema, seed the asset universe and optionally load real historical data."""

import argparse

from backend.app.database import Base, SessionLocal, engine
from src.data.ingestion import ingest_history, seed_assets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--symbols", nargs="*")
    args = parser.parse_args()
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        print(f"Seeded {seed_assets(db)} assets")
        if not args.skip_download:
            print(ingest_history(db, years=args.years, symbols=args.symbols))


if __name__ == "__main__":
    main()
