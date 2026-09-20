from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

REGIME_FEATURES = [
    "return_20d",
    "volatility_20",
    "momentum_10",
    "relative_volume_20",
    "drawdown_60",
]


@dataclass
class RegimeResult:
    labels: pd.Series
    descriptions: dict[int, str]
    metrics: dict[str, float]
    transition_matrix: list[list[float]] | None = None
    model: object | None = None
    scaler: StandardScaler | None = None


def characterize(frame: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    descriptions = {}
    work = frame.copy()
    work["state"] = labels
    med_vol = work["volatility_20"].median()
    for state, group in work.groupby("state"):
        ret, vol, dd, momentum = (
            group["return_1d"].mean(),
            group["volatility_20"].mean(),
            group["drawdown_60"].mean(),
            group["momentum_10"].mean(),
        )
        direction = (
            "positive-return"
            if ret > 0.0003
            else "negative-return"
            if ret < -0.0003
            else "range-bound"
        )
        risk = "high-volatility" if vol > med_vol else "lower-volatility"
        trend = "positive momentum" if momentum > 0 else "negative momentum"
        descriptions[int(state)] = (
            f"State {state}: {direction}, {risk}, {trend}; mean drawdown {dd:.1%}"
        )
    return descriptions


def discover_regimes(
    frame: pd.DataFrame, algorithm: str = "gmm", n_states: int = 3, seed: int = 42
) -> RegimeResult:
    clean = frame.dropna(subset=REGIME_FEATURES).copy()
    if len(clean) < max(80, n_states * 20):
        raise ValueError("Insufficient complete observations for regime discovery")
    scaler = StandardScaler()
    x = scaler.fit_transform(clean[REGIME_FEATURES])
    transition = None
    if algorithm == "kmeans":
        model = KMeans(n_clusters=n_states, n_init=20, random_state=seed)
        labels = model.fit_predict(x)
    elif algorithm == "gmm":
        model = GaussianMixture(
            n_components=n_states, covariance_type="full", n_init=10, random_state=seed
        )
        labels = model.fit_predict(x)
    elif algorithm == "hmm":
        model = GaussianHMM(
            n_components=n_states, covariance_type="full", n_iter=300, random_state=seed
        )
        model.fit(x)
        labels = model.predict(x)
        transition = model.transmat_.tolist()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    metrics = {
        "silhouette": float(silhouette_score(x, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(x, labels)),
    }
    series = pd.Series(labels, index=clean.index, name="regime")
    return RegimeResult(series, characterize(clean, labels), metrics, transition, model, scaler)
