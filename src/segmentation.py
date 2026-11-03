"""
Customer Segmentation Using K-Means Clustering
==============================================
Data Mining mini-project - end-to-end pipeline.

"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: figures are saved, not displayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
DEFAULT_FEATURES = ["Age", "Annual_Income_k", "Spending_Score"]
ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("segmentation")
sns.set_theme(style="whitegrid", context="notebook")
PALETTE = sns.color_palette("Set2", 10)


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #
def load_data(path: Path) -> pd.DataFrame:
    """Load the CSV and run basic integrity checks."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python data/generate_dataset.py` first."
        )
    df = pd.read_csv(path)
    log.info("Loaded %d rows x %d columns from %s", *df.shape, path)
    if df.duplicated("CustomerID").any():
        log.warning("Duplicate CustomerID values detected - dropping duplicates")
        df = df.drop_duplicates("CustomerID")
    return df


# --------------------------------------------------------------------------- #
# 2. Exploratory data analysis
# --------------------------------------------------------------------------- #
def run_eda(df: pd.DataFrame, features: list[str], fig_dir: Path) -> None:
    log.info("Missing values per column:\n%s", df.isna().sum().to_string())
    log.info("Descriptive statistics:\n%s", df[features].describe().round(2).to_string())

    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4))
    for ax, col in zip(np.atleast_1d(axes), features):
        sns.histplot(df[col], kde=True, ax=ax, color=PALETTE[0])
        ax.set_title(f"Distribution of {col}")
    fig.tight_layout()
    fig.savefig(fig_dir / "01_distributions.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", ax=ax)
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    fig.savefig(fig_dir / "02_correlation.png", dpi=150)
    plt.close(fig)

    if "Gender" in df.columns:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=df, x="Gender", hue="Gender", palette="Set2", legend=False, ax=ax)
        ax.set_title("Customers by gender")
        fig.tight_layout()
        fig.savefig(fig_dir / "03_gender.png", dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------- #
# 3. Preprocessing
# --------------------------------------------------------------------------- #
def preprocess(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, StandardScaler]:
    """Standardize features (mean 0, std 1). K-Means is distance based, so
    features on larger scales would otherwise dominate the clustering."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    return X, scaler


# --------------------------------------------------------------------------- #
# 4. Choosing k
# --------------------------------------------------------------------------- #
def evaluate_k_range(X: np.ndarray, k_min: int, k_max: int) -> pd.DataFrame:
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        rows.append(
            {
                "k": k,
                "inertia_wcss": km.inertia_,
                "silhouette": silhouette_score(X, km.labels_),
                "davies_bouldin": davies_bouldin_score(X, km.labels_),
                "calinski_harabasz": calinski_harabasz_score(X, km.labels_),
            }
        )
    return pd.DataFrame(rows)


def plot_k_selection(metrics: pd.DataFrame, best_k: int, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    spec = [
        ("inertia_wcss", "Elbow method (WCSS / inertia)", "WCSS"),
        ("silhouette", "Silhouette score (higher = better)", "Silhouette"),
        ("davies_bouldin", "Davies-Bouldin index (lower = better)", "Davies-Bouldin"),
    ]
    for ax, (col, title, ylabel) in zip(axes, spec):
        ax.plot(metrics["k"], metrics[col], marker="o", color=PALETTE[0], lw=2)
        ax.axvline(best_k, ls="--", color="crimson", label=f"chosen k = {best_k}")
        ax.set_title(title)
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel(ylabel)
        ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "04_choosing_k.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5-6. Final model and profiling
# --------------------------------------------------------------------------- #
def fit_final_model(X: np.ndarray, k: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X)


def profile_clusters(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Per-cluster mean of each feature (original units), size and share."""
    prof = df.groupby("Cluster")[features].mean().round(1)
    prof["Customers"] = df.groupby("Cluster").size()
    prof["Share_%"] = (prof["Customers"] / len(df) * 100).round(1)
    if "Gender" in df.columns:
        prof["Female_%"] = (
            df.assign(f=df["Gender"].eq("Female")).groupby("Cluster")["f"].mean() * 100
        ).round(1)
    return prof


def label_clusters(prof: pd.DataFrame) -> dict[int, str]:
    """Rule-based business names: each cluster is compared with the median
    income / spending across clusters. Duplicate names get a numeric suffix."""
    if not {"Annual_Income_k", "Spending_Score"}.issubset(prof.columns):
        return {c: f"Segment {c}" for c in prof.index}

    inc_med = prof["Annual_Income_k"].median()
    spd_med = prof["Spending_Score"].median()
    age_med = prof["Age"].median() if "Age" in prof.columns else None
    names: dict[int, str] = {}
    for c, r in prof.iterrows():
        hi_inc = r["Annual_Income_k"] > inc_med
        hi_spd = r["Spending_Score"] > spd_med
        if hi_inc and hi_spd:
            name = "Premium / VIP"
        elif hi_inc and not hi_spd:
            name = "Cautious Affluent"
        elif not hi_inc and hi_spd:
            name = "Young Enthusiasts" if age_med and r["Age"] < age_med else "Impulsive Spenders"
        elif r["Annual_Income_k"] == inc_med and r["Spending_Score"] == spd_med:
            name = "Average Customers"
        else:
            name = "Budget-Conscious"
        names[c] = name

    totals = pd.Series(names).value_counts()
    counter: dict[str, int] = {}
    for c in list(names):
        n = names[c]
        if totals[n] > 1:
            counter[n] = counter.get(n, 0) + 1
            names[c] = f"{n} ({counter[n]})"
    return names


# --------------------------------------------------------------------------- #
# 7. Visualization
# --------------------------------------------------------------------------- #
def plot_scatter_2d(df: pd.DataFrame, km: KMeans, scaler: StandardScaler,
                    features: list[str], fig_dir: Path) -> None:
    if not {"Annual_Income_k", "Spending_Score"}.issubset(features):
        return
    centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=features)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sns.scatterplot(data=df, x="Annual_Income_k", y="Spending_Score", hue="Segment",
                    palette="Set2", s=60, edgecolor="white", ax=ax)
    ax.scatter(centers["Annual_Income_k"], centers["Spending_Score"], c="black",
               s=260, marker="X", edgecolor="white", linewidth=1.5, label="Centroids")
    ax.set_title("Customer segments: Annual Income vs Spending Score")
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "05_segments_income_vs_spending.png", dpi=150)
    plt.close(fig)


def plot_pca(df: pd.DataFrame, X: np.ndarray, fig_dir: Path) -> float:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    comp = pca.fit_transform(X)
    var = pca.explained_variance_ratio_.sum() * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=comp[:, 0], y=comp[:, 1], hue=df["Segment"], palette="Set2",
                    s=55, edgecolor="white", ax=ax)
    ax.set_title(f"PCA projection of segments ({var:.1f}% variance explained)")
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "06_pca_projection.png", dpi=150)
    plt.close(fig)
    return var


def plot_silhouette(X: np.ndarray, labels: np.ndarray, fig_dir: Path) -> None:
    k = len(np.unique(labels))
    sil = silhouette_samples(X, labels)
    avg = sil.mean()
    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10
    for i in range(k):
        vals = np.sort(sil[labels == i])
        y_upper = y_lower + len(vals)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, color=PALETTE[i % 8], alpha=0.85)
        ax.text(-0.05, y_lower + 0.5 * len(vals), f"C{i}")
        y_lower = y_upper + 10
    ax.axvline(avg, color="crimson", ls="--", label=f"average = {avg:.3f}")
    ax.set_title("Silhouette plot of the final clustering")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Customers (grouped by cluster)")
    ax.set_yticks([])
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "07_silhouette_plot.png", dpi=150)
    plt.close(fig)


def plot_profiles(prof: pd.DataFrame, names: dict[int, str], features: list[str], fig_dir: Path) -> None:
    long = (prof[features].rename(index=names).rename_axis("Cluster").reset_index()
            .melt(id_vars="Cluster", var_name="Feature", value_name="Mean"))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=long, x="Feature", y="Mean", hue="Cluster", palette="Set2", ax=ax)
    ax.set_title("Average feature values per segment")
    ax.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "08_segment_profiles.png", dpi=150)
    plt.close(fig)


def plot_sizes(df: pd.DataFrame, fig_dir: Path) -> None:
    counts = df["Segment"].value_counts()
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90,
           colors=PALETTE[: len(counts)], wedgeprops={"edgecolor": "white"})
    ax.set_title("Segment size distribution")
    fig.tight_layout()
    fig.savefig(fig_dir / "09_segment_sizes.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "customers.csv")
    p.add_argument("--out", type=Path, default=ROOT / "outputs")
    p.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    p.add_argument("--k", type=int, default=None, help="force k (default: best silhouette)")
    p.add_argument("--min-k", type=int, default=2)
    p.add_argument("--max-k", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fig_dir = args.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    missing = [f for f in args.features if f not in df.columns]
    if missing:
        raise ValueError(f"Unknown feature(s): {missing}. Available: {list(df.columns)}")
    df = df.dropna(subset=args.features).reset_index(drop=True)

    run_eda(df, args.features, fig_dir)
    X, scaler = preprocess(df, args.features)

    metrics = evaluate_k_range(X, args.min_k, args.max_k)
    best_k = args.k or int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
    log.info("Model selection metrics:\n%s", metrics.round(3).to_string(index=False))
    log.info("Selected k = %d", best_k)
    plot_k_selection(metrics, best_k, fig_dir)

    km = fit_final_model(X, best_k)
    df["Cluster"] = km.labels_
    prof = profile_clusters(df, args.features)
    names = label_clusters(prof)
    df["Segment"] = df["Cluster"].map(names)
    prof.insert(0, "Segment", prof.index.map(names))
    log.info("Cluster profiles:\n%s", prof.to_string())

    plot_scatter_2d(df, km, scaler, args.features, fig_dir)
    var = plot_pca(df, X, fig_dir)
    plot_silhouette(X, km.labels_, fig_dir)
    plot_profiles(prof.drop(columns="Segment"), names, args.features, fig_dir)
    plot_sizes(df, fig_dir)

    final = {
        "k": best_k,
        "silhouette": silhouette_score(X, km.labels_),
        "davies_bouldin": davies_bouldin_score(X, km.labels_),
        "calinski_harabasz": calinski_harabasz_score(X, km.labels_),
        "wcss": km.inertia_,
        "pca_2d_variance_%": var,
    }
    metrics.round(4).to_csv(args.out / "k_selection_metrics.csv", index=False)
    pd.Series(final).round(4).to_csv(args.out / "final_metrics.csv", header=["value"])
    prof.to_csv(args.out / "cluster_profiles.csv")
    df.to_csv(args.out / "customers_segmented.csv", index=False)
    log.info("Final metrics: %s", {k: round(float(v), 3) for k, v in final.items()})
    log.info("All outputs saved in %s", args.out)


if __name__ == "__main__":
    main()
