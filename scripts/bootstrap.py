"""Initialize schema, seed the asset universe and optionally load real historical data."""

import argparse

from backend.app.database import Base, SessionLocal, engine
from src.data.ingestion import ingest_history, seed_assets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--strict", action="store_true", help="Fail when any provider symbol fails")
    args = parser.parse_args()
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        print(f"Seeded {seed_assets(db)} assets")
        if not args.skip_download:
            result = ingest_history(db, years=args.years, symbols=args.symbols)
            print(result)
            if args.strict and result["failures"]:
                raise SystemExit("Market-data bootstrap was incomplete; rerun setup when the provider is available")


if __name__ == "__main__":
    main()
