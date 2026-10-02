"""Descriptive K-Means grouping, fitted separately inside each stock sector."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.features.pipeline import build_features

CLUSTER_FEATURES = ["return_20d", "volatility_20", "sma_ratio_20", "rsi_14", "relative_volume_20", "drawdown_60"]


def describe_stock(prices: pd.DataFrame) -> dict | None:
    if len(prices) < 80:
        return None
    row = build_features(prices).iloc[-1]
    if any(pd.isna(row.get(name)) for name in CLUSTER_FEATURES):
        return None
    return {name: float(row[name]) for name in CLUSTER_FEATURES}


def cluster_stocks(observations: dict[str, pd.DataFrame], sectors: dict[str, str | None]) -> dict:
    rows = {symbol: describe_stock(frame) for symbol, frame in observations.items()}
    rows = {symbol: item for symbol, item in rows.items() if item is not None and sectors.get(symbol)}
    if not rows:
        return {"status": "insufficient_data", "message": "At least 80 historical sessions are required.", "stocks": {}}
    stock_results, groups = {}, {}
    for sector in sorted(set(sectors[s] for s in rows)):
        symbols = sorted(s for s in rows if sectors[s] == sector)
        if len(symbols) < 4:
            continue
        matrix = np.array([[rows[s][name] for name in CLUSTER_FEATURES] for s in symbols])
        scaled = StandardScaler().fit_transform(matrix)
        # Partition only when each resulting group has at least three members.
        labels = KMeans(n_clusters=2, random_state=42, n_init=20).fit_predict(scaled) if len(symbols) >= 6 else np.zeros(len(symbols), dtype=int)
        if min(np.bincount(labels)) < 3:
            labels = np.zeros(len(symbols), dtype=int)
        count = len(set(labels))
        silhouette = (float(silhouette_score(scaled, labels))
                      if 1 < count < len(symbols) and len(set(labels)) > 1 else None)
        for group in range(count):
            members = np.flatnonzero(labels == group)
            means = matrix[members].mean(axis=0)
            summary = (f"Average 20-session return {means[0]:+.1%}; annualized volatility "
                       f"{means[1]:.1%}; average drawdown {means[5]:.1%}.")
            key = f"{sector}:{group}"
            groups[key] = {"name": f"Similar {sector} stocks", "sector": sector,
                           "count": len(members), "description": summary,
                           "average_20d_return": float(means[0]),
                           "average_annualized_volatility": float(means[1]),
                           "average_drawdown": float(means[5]), "silhouette_score": silhouette,
                           "clusters_in_sector": count}
            for index in members:
                peers = sorted((i for i in members if i != index),
                               key=lambda i: float(np.linalg.norm(scaled[i] - scaled[index])))
                symbol = symbols[index]
                stock_results[symbol] = {"cluster_id": int(group), "cluster": groups[key],
                                         "sector": sector, "characteristics": rows[symbol],
                                         "similar_stocks": [{"symbol": symbols[i], "sector": sector,
                                                             "distance": float(np.linalg.norm(scaled[i] - scaled[index]))}
                                                            for i in peers[:5]]}
    if not stock_results:
        return {"status": "insufficient_data", "message": "Not enough historical data to identify a reliable peer group.", "stocks": {}}
    return {"status": "ok", "method": "Sector-constrained K-Means on standardized real-stock characteristics; descriptive, not predictive",
            "stocks": stock_results, "groups": groups}
