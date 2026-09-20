import numpy as np

FRIENDLY = {
    "return_1d": "the latest daily return",
    "return_20d": "the medium-term return",
    "rsi_14": "momentum (RSI)",
    "macd": "trend momentum (MACD)",
    "relative_volume_20": "trading volume versus its recent norm",
    "volatility_20": "recent volatility",
    "regime": "the detected market regime",
    "sentiment_mean": "recent financial-news sentiment",
    "anomaly_score": "unusual market activity",
}


def explain_contributions(
    feature_names: list[str],
    values: list[float],
    probability_up: float,
    top_n: int = 4,
) -> str:
    pairs = sorted(zip(feature_names, values, strict=True), key=lambda p: abs(p[1]), reverse=True)[
        :top_n
    ]
    positive = [FRIENDLY.get(name, name.replace("_", " ")) for name, value in pairs if value > 0]
    negative = [FRIENDLY.get(name, name.replace("_", " ")) for name, value in pairs if value < 0]
    direction = "UP" if probability_up >= 0.5 else "DOWN"
    parts = [
        f"The model predicts {direction} with P(UP)={probability_up:.0%} and "
        f"P(DOWN)={1 - probability_up:.0%}."
    ]
    if positive:
        parts.append("Upward contribution came mainly from " + ", ".join(positive) + ".")
    if negative:
        parts.append("Downward contribution came mainly from " + ", ".join(negative) + ".")
    parts.append("These are model attributions, not causal explanations or investment advice.")
    return " ".join(parts)


def tree_shap(model: object, x, feature_names: list[str]) -> dict:
    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return {"features": feature_names, "values": values[0].tolist()}
