"""Inspect development-period synthetic classification without touching the final test."""


from src.ml.final_prediction import development_split, load_research_panel
from src.ml.research import metrics
from src.ml.synthetic_research import FEATURE_GROUPS, MODEL_CANDIDATES, make_synthetic_model


def main() -> None:
    frame = load_research_panel()
    print("rows", len(frame), "assets", frame.symbol.nunique(), "duplicate asset-times", frame.duplicated(["symbol", "timestamp"]).sum())
    for horizon in (1, 3, 5):
        train, validation, _ = development_split(frame, horizon)
        print("horizon", horizon, "train", len(train), "validation", len(validation), "UP", round(float(validation[f"target_{horizon}d"].mean()), 3))
        for feature_group in ("technical_market", "technical_market_regime_volume"):
            features = FEATURE_GROUPS[feature_group]
            for name in ("logistic_regression", "xgboost"):
                for parameters in MODEL_CANDIDATES[name]:
                    model = make_synthetic_model(name, parameters)
                    model.fit(train[features], train[f"target_{horizon}d"].astype(int))
                    probability = model.predict_proba(validation[features])[:, 1]
                    score = metrics(validation[f"target_{horizon}d"], probability)
                    print(horizon, feature_group, name, parameters.get("C", parameters.get("max_depth")), round(score["accuracy"], 4), round(score["balanced_accuracy"], 4), round(score["roc_auc"], 4), flush=True)


if __name__ == "__main__":
    main()
