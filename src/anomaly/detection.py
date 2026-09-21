import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

ANOMALY_FEATURES = ["return_1d", "volatility_20", "relative_volume_20", "volume_change"]


def detect_anomalies(
    frame: pd.DataFrame, contamination: float = 0.025, seed: int = 42
) -> pd.DataFrame:
    clean = frame.dropna(subset=ANOMALY_FEATURES).copy()
    if len(clean) < 50:
        raise ValueError("At least 50 complete rows are required for anomaly detection")
    x = StandardScaler().fit_transform(clean[ANOMALY_FEATURES])
    iso = IsolationForest(contamination=contamination, random_state=seed).fit(x)
    lof = LocalOutlierFactor(n_neighbors=min(20, len(clean) - 1), contamination=contamination)
    iso_score = -iso.decision_function(x)
    lof_label = lof.fit_predict(x)
    lof_score = -lof.negative_outlier_factor_
    ensemble_rank = (
        pd.Series(iso_score).rank(pct=True).to_numpy()
        + pd.Series(lof_score).rank(pct=True).to_numpy()
    ) / 2
    # The old score was an average of two ranks, not itself a percentile.
    # Re-rank the ensemble across ALL scored observations, not only flagged ones.
    clean["anomaly_score"] = ensemble_rank
    clean["unusualness_percentile"] = pd.Series(ensemble_rank).rank(pct=True).to_numpy() * 100
    clean["isolation_raw"] = iso_score
    clean["lof_raw"] = lof_score
    clean["is_anomaly"] = (iso.predict(x) == -1) | (lof_label == -1)
    magnitudes = pd.DataFrame(
        {
            "abnormal_return": clean["return_1d"].abs(),
            "unusual_volume": (clean["relative_volume_20"] - 1).abs(),
            "volatility_spike": clean["volatility_20"],
            "volume_change": clean["volume_change"].abs(),
        },
        index=clean.index,
    )
    robust_scale = magnitudes.median().replace(0, np.nan)
    clean["anomaly_type"] = magnitudes.div(robust_scale).fillna(0).idxmax(axis=1)
    clean["severity"] = pd.cut(
        clean["unusualness_percentile"],
        bins=[-np.inf, 95, 99, np.inf],
        labels=["low", "medium", "high"],
    ).astype(str)
    return clean
