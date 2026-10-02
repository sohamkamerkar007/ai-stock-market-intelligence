"""Development-only stability check across seeded controlled panels."""

import argparse

from scripts.train_final_classifiers import PARAMETERS
from src.ml.final_prediction import FINAL_FEATURES, development_split
from src.ml.research import metrics
from src.ml.synthetic_research import (
    SyntheticSpec,
    build_synthetic_features,
    generate_synthetic_market,
    make_synthetic_model,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gain", type=float, required=True)
    args = parser.parse_args()
    frame = build_synthetic_features(generate_synthetic_market(SyntheticSpec(seed=args.seed, signal_gain=args.gain)))
    train, validation, _ = development_split(frame, 3)
    print("seed", args.seed, "gain", args.gain, "validation UP", round(float(validation.target_3d.mean()), 3))
    for name, params in PARAMETERS.items():
        model = make_synthetic_model(name, params)
        model.fit(train[FINAL_FEATURES], train.target_3d.astype(int))
        result = metrics(validation.target_3d, model.predict_proba(validation[FINAL_FEATURES])[:, 1])
        print(name, "validation accuracy", round(result["accuracy"], 4), "balanced", round(result["balanced_accuracy"], 4), "AUC", round(result["roc_auc"], 4))


if __name__ == "__main__":
    main()
