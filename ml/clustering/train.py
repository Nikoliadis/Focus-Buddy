"""KMeans clustering of user behaviour.

Steps:
1. Standardise every feature (KMeans uses Euclidean distance: without scaling,
   ``sessions_per_week`` (0-20) would drown out shares (0-1)).
2. For k = 2..8, fit KMeans and compute the silhouette score and the inertia.
3. Pick k with the best silhouette, then check that the clustering is *stable*:
   refit on bootstrap resamples and compare labels with the adjusted Rand index.
4. Report profiles in the original units (see ml.clustering.analyze).

KMeans always returns k clusters, even for data without any structure. The silhouette
and stability checks are what tell us whether the clusters mean anything.

Usage:
    python -m ml.clustering.train --source synthetic
    python -m ml.clustering.train --source db
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.clustering.features import USER_FEATURES, build_user_features
from ml.data import InsufficientDataError, load_logs_rooms

DEFAULT_K_RANGE = range(2, 9)
MIN_USERS = 20
# Common rule of thumb (Kaufman & Rousseeuw): below ~0.25 there is no substantial structure.
WEAK_SILHOUETTE = 0.25
UNSTABLE_ARI = 0.6
DEFAULT_ARTIFACT_DIR = Path("artifacts/clustering")


def make_pipeline(k: int, random_state: int = 0) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=20, random_state=random_state)),
    ])


def evaluate_k(X: pd.DataFrame, k_range=DEFAULT_K_RANGE, random_state: int = 0) -> pd.DataFrame:
    """Silhouette (higher is better, in [-1, 1]) and inertia (within-cluster sum of squares,
    always decreases with k, so only its "elbow" is informative) for each k."""
    rows = []
    Z = StandardScaler().fit_transform(X)
    for k in k_range:
        if k >= len(X):
            break
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state).fit(Z)
        rows.append({"k": k, "silhouette": float(silhouette_score(Z, km.labels_)), "inertia": float(km.inertia_)})
    return pd.DataFrame(rows)


def stability(X: pd.DataFrame, k: int, n_runs: int = 20, random_state: int = 0) -> float:
    """Mean adjusted Rand index between the full-data labels and labels from models fitted
    on bootstrap resamples. 1 = identical partitions; ~0 = no better than random."""
    rng = np.random.default_rng(random_state)
    reference = make_pipeline(k, random_state).fit(X).predict(X)
    scores = []
    for run in range(n_runs):
        sample = X.iloc[rng.integers(0, len(X), len(X))]
        labels = make_pipeline(k, random_state + run + 1).fit(sample).predict(X)
        scores.append(adjusted_rand_score(reference, labels))
    return float(np.mean(scores))


@dataclass
class ClusteringReport:
    data_source: str
    n_users: int
    excluded_users: int
    k: int
    k_scores: list[dict]
    silhouette: float
    stability_ari: float
    sizes: dict[int, int]
    features: list[str] = field(default_factory=lambda: list(USER_FEATURES))
    trained_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    @property
    def weak_structure(self) -> bool:
        return self.silhouette < WEAK_SILHOUETTE or self.stability_ari < UNSTABLE_ARI


def fit_clusters(
    user_features: pd.DataFrame,
    data_source: str,
    k: int | None = None,
    excluded_users: int = 0,
    random_state: int = 0,
) -> tuple[ClusteringReport, Pipeline, pd.Series]:
    """Return the report, the fitted scaler+KMeans pipeline and each user's cluster."""
    X = user_features[list(USER_FEATURES)]
    if len(X) < MIN_USERS:
        raise InsufficientDataError(f"need >= {MIN_USERS} users with enough sessions, got {len(X)}")
    if X.isna().any().any():
        raise ValueError("user features contain missing values")

    scores = evaluate_k(X, random_state=random_state)
    chosen = k if k is not None else int(scores.loc[scores["silhouette"].idxmax(), "k"])
    model = make_pipeline(chosen, random_state).fit(X)
    labels = pd.Series(model.predict(X), index=X.index, name="cluster")

    report = ClusteringReport(
        data_source=data_source, n_users=len(X), excluded_users=excluded_users, k=chosen,
        k_scores=scores.to_dict("records"),
        silhouette=float(silhouette_score(model.named_steps["scale"].transform(X), labels)),
        stability_ari=stability(X, chosen, random_state=random_state),
        sizes={int(c): int(n) for c, n in labels.value_counts().sort_index().items()},
    )
    return report, model, labels


def main(argv: list[str] | None = None) -> int:
    import argparse

    from ml import artifacts
    from ml.clustering import analyze

    parser = argparse.ArgumentParser(description="Cluster users by focus behaviour.")
    parser.add_argument("--source", choices=["synthetic", "db"], required=True)
    parser.add_argument("--k", type=int, default=None, help="fix k instead of choosing by silhouette")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)

    logs, rooms = load_logs_rooms(args.source, args.seed)
    all_users = logs["user_id"].nunique()
    users = build_user_features(logs, rooms)
    try:
        report, model, labels = fit_clusters(users, args.source, args.k, all_users - len(users), args.seed)
    except InsufficientDataError as exc:
        print(f"Not clustering: {exc}")
        return 2

    print(analyze.report_text(report, users, labels))
    if not args.no_save:
        stamp = report.trained_at.replace(":", "").replace("-", "")
        meta = {**asdict(report), "features": list(USER_FEATURES)}
        # artifacts.load checks the *session* feature list; clustering models are loaded with joblib directly.
        path = artifacts.save(model, meta, args.out, f"kmeans_k{report.k}_{report.data_source}_{stamp}")
        fig = analyze.profile_heatmap(users, labels)
        fig.savefig(path.with_suffix(".png"))
        print(f"\nsaved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
