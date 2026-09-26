"""Generate, train and persist the isolated controlled synthetic research experiment."""

import argparse

from src.ml.synthetic_research import (
    SyntheticSpec,
    build_synthetic_features,
    generate_synthetic_market,
    run_synthetic_experiment,
    save_synthetic_outputs,
    spec_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--sessions", type=int, default=2200)
    parser.add_argument("--assets-per-sector", type=int, default=3)
    args = parser.parse_args()
    spec = SyntheticSpec(
        seed=args.seed,
        sessions=args.sessions,
        assets_per_sector=args.assets_per_sector,
    )
    raw = generate_synthetic_market(spec)
    frame = build_synthetic_features(raw, threshold=spec.target_threshold)
    report, artifact = run_synthetic_experiment(frame)
    report["generator"] = spec_metadata(spec)
    paths = save_synthetic_outputs(frame, report, artifact)
    print("Synthetic research dataset generated", flush=True)
    print(f"Observations: {report['dataset']['observations']}", flush=True)
    print(f"Assets: {report['dataset']['assets']}; sectors: {report['dataset']['sectors']}", flush=True)
    print(f"Selected: {report['selected']['model']} / {report['selected']['feature_group']} / {report['target']['horizon_sessions']} session", flush=True)
    print(f"Final test metrics: {report['selected']['test']}", flush=True)
    print(f"Outputs: {paths}", flush=True)


if __name__ == "__main__":
    main()
