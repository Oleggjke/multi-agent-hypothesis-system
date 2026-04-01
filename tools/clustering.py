import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

def run_kmeans(df: pd.DataFrame, features: list, n_clusters: int = 3) -> dict:
    clean = df[features].dropna()

    # масштабируем данные
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean)

    # запускаем кластеризацию
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    # считаем силуэтный коэффициент
    sil_score = silhouette_score(X_scaled, labels) if n_clusters > 1 else 0

    # размеры кластеров
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip([int(u) for u in unique], [int(c) for c in counts]))

    return {
        "method": "kmeans",
        "features": features,
        "n_clusters": n_clusters,
        "silhouette_score": round(float(sil_score), 4),
        "cluster_sizes": cluster_sizes,
        "inertia": round(float(model.inertia_), 4),
        "n": len(clean)
    }
