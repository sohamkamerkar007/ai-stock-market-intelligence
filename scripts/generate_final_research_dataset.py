"""Generate the separately seeded controlled final prediction panel."""

from pathlib import Path

from src.ml.synthetic_research import (
    SyntheticSpec,
    build_synthetic_features,
    generate_synthetic_market,
)


def main() -> None:
    # Chosen on two development-only seeds, before evaluating this independent
    # final seed. The earlier 20261001 run was retired as exploratory.
    spec = SyntheticSpec(seed=20261004, signal_gain=2.0)
    frame = build_synthetic_features(generate_synthetic_market(spec))
    path = Path("data/processed/final_prediction_research.csv.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip")
    print("Saved", len(frame), "rows,", frame.symbol.nunique(), "assets to", path)


if __name__ == "__main__":
    main()
