from src.explainability.shap_explanations import explain_contributions


def test_human_explanation_is_grounded():
    text = explain_contributions(["rsi_14", "volatility_20"], [0.3, -0.2], 0.63)
    assert "63%" in text
    assert "P(DOWN)=37%" in text
    assert "momentum" in text
    assert "volatility" in text
    assert "not causal" in text


def test_down_explanation_keeps_positive_class_shap_semantics():
    text = explain_contributions(["macd"], [0.2], 0.4)
    assert "predicts DOWN" in text
    assert "Upward contribution" in text
