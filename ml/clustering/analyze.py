"""Describe clusters with measurable characteristics only.

A cluster is "users whose evening share is 1.4 SD above average", never a persona such as
"night owls" or "procrastinators": the data shows behaviour, not motives or personality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from ml.clustering.features import USER_FEATURES

SHARE_FEATURES = {c for c in USER_FEATURES if c.endswith(("_share", "_rate"))}


def cluster_profiles(user_features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Per cluster: size and the mean of every feature, in original units, plus an overall row."""
    X = user_features[list(USER_FEATURES)]
    by_cluster = X.groupby(labels).mean()
    by_cluster.insert(0, "users", labels.value_counts().sort_index())
    overall = X.mean().to_frame("all").T
    overall.insert(0, "users", len(X))
    out = pd.concat([by_cluster, overall])
    out.index = [f"cluster {i}" if i != "all" else "all users" for i in out.index]
    return out


def standardised_differences(user_features: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """(cluster mean - overall mean) / overall SD for each feature: how unusual each cluster is."""
    X = user_features[list(USER_FEATURES)]
    sd = X.std(ddof=1).replace(0, np.nan)
    return (X.groupby(labels).mean() - X.mean()) / sd


def _fmt(feature: str, value: float) -> str:
    return f"{value:.0%}" if feature in SHARE_FEATURES else f"{value:.1f}"


def describe(user_features: pd.DataFrame, labels: pd.Series, top: int = 3, min_sd: float = 0.5) -> dict[int, str]:
    """One sentence per cluster from its most distinctive features (|difference| >= min_sd)."""
    diffs = standardised_differences(user_features, labels)
    X = user_features[list(USER_FEATURES)]
    means, overall = X.groupby(labels).mean(), X.mean()
    out = {}
    for cluster, row in diffs.iterrows():
        strongest = row.abs().sort_values(ascending=False)
        parts = []
        for feature in strongest.index[:top]:
            if strongest[feature] < min_sd:
                break
            direction = "higher" if row[feature] > 0 else "lower"
            parts.append(f"{direction} {feature.replace('_', ' ')} ({_fmt(feature, means.loc[cluster, feature])} "
                         f"vs {_fmt(feature, overall[feature])} overall, {row[feature]:+.1f} SD)")
        n = int((labels == cluster).sum())
        out[int(cluster)] = f"{n} users: " + ("; ".join(parts) if parts else "close to the overall average")
    return out


def report_text(report, user_features: pd.DataFrame, labels: pd.Series) -> str:
    banner = "=" * 76
    lines = []
    if report.data_source != "db":
        lines += [banner, f"  {report.data_source.upper()} DATA: these clusters say nothing about real users", banner]
    lines += [
        f"Users clustered: {report.n_users} (excluded {report.excluded_users} with too few sessions)",
        "",
        "  k   silhouette    inertia",
    ]
    for r in report.k_scores:
        chosen = "  <- chosen" if r["k"] == report.k else ""
        lines.append(f"{r['k']:>3}   {r['silhouette']:>10.3f} {r['inertia']:>10.1f}{chosen}")
    lines += [
        "",
        f"k = {report.k}: silhouette {report.silhouette:.3f}, bootstrap stability (ARI) {report.stability_ari:.3f}",
    ]
    if report.weak_structure:
        lines.append("WARNING: weak or unstable structure: treat these clusters as a convenience split, "
                     "not as distinct user groups.")
    lines += ["", "Clusters (measurable differences from the average user):"]
    lines += [f"  cluster {c}: {text}" for c, text in describe(user_features, labels).items()]
    lines += ["", cluster_profiles(user_features, labels).round(2).to_string()]
    return "\n".join(lines)


def profile_heatmap(user_features: pd.DataFrame, labels: pd.Series) -> Figure:
    """Features × clusters, coloured by (cluster mean - overall mean) / user-level SD.

    Diverging scale: blue = below average, red = above, neutral grey = average. The SD is
    taken across *users*, not across cluster means: with two clusters the latter would
    turn every feature into exactly ±1 and make trivial differences look as strong as real ones."""
    from analytics.charts import LIGHT, _style, _title

    profiles = cluster_profiles(user_features, labels)
    clusters = profiles.drop(index="all users").drop(columns="users")
    z = standardised_differences(user_features, labels).fillna(0)
    z.index = clusters.index
    z = z.T

    fig = Figure(figsize=(1.3 * len(clusters) + 4, 0.45 * len(z) + 1.6), dpi=110,
                 facecolor=LIGHT.surface, layout="constrained")
    ax = fig.subplots()
    cmap = LinearSegmentedColormap.from_list("diverging", ["#2a78d6", "#f0efec", "#e34948"])
    limit = max(1.0, float(np.nanmax(np.abs(z.to_numpy()))))
    mesh = ax.imshow(z.to_numpy(), cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(z.columns)), [f"{c}\n({int(profiles.loc[c, 'users'])} users)" for c in z.columns])
    ax.set_yticks(range(len(z.index)), [f.replace("_", " ") for f in z.index])
    for (i, j), value in np.ndenumerate(clusters.T.to_numpy()):
        feature = z.index[i]
        ax.text(j, i, _fmt(feature, value), ha="center", va="center", fontsize=8, color=LIGHT.ink)
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.set_label("Difference from average user (SD)", color=LIGHT.ink_secondary)
    _style(ax, LIGHT, grid_axis="")
    for spine in ax.spines.values():
        spine.set_visible(False)
    _title(ax, LIGHT, "Cluster profiles", "Cell text = cluster mean; colour = how far it is from the average user")
    return fig
