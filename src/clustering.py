import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

# Headless / CI-friendly rendering
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import seaborn as sns  # noqa: E402

from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402


# --- Global styling for publication-ready figures (IEEE 2-column legibility) ---
sns.set_context("paper")
sns.set_style("whitegrid")
plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "lines.linewidth": 3.0,
        "lines.markersize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def _kmeans_labels(x_scaled: np.ndarray, k: int, seed: int) -> np.ndarray:
    km = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=1,
        max_iter=300,
        random_state=seed,
    )
    return km.fit_predict(x_scaled)


def _kmeans_fit_for_metrics(x_scaled: np.ndarray, k: int, seed: int):
    km = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=1,
        max_iter=300,
        random_state=seed,
    )
    labels = km.fit_predict(x_scaled)
    inertia = float(km.inertia_)
    return inertia, labels


def compute_cluster_profile(df: pd.DataFrame) -> pd.DataFrame:
    # Interpretive cluster labeling uses these three features.
    required = ["area_m2", "density_ratio", "orientation_entropy"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for interpretive profiling: {missing}")

    profile = (
        df.groupby("cluster_id", as_index=False)[required]
        .mean()
        .rename(
            columns={
                "area_m2": "mean_area_m2",
                "density_ratio": "mean_density_ratio",
                "orientation_entropy": "mean_orientation_entropy",
            }
        )
    )

    # Risk scoring rule:
    # - cluster with highest entropy and density ratio => Informal / High Risk
    # - cluster with lowest entropy and largest area => Stable / Formal Infrastructure
    # - remaining => Planned Residential
    # NOTE: we implement exactly what the directive describes using explicit ranking.
    # "largest area" = highest mean_area_m2
    # "lowest entropy" = lowest mean_orientation_entropy
    #
    # Informal/high-risk selection (matches directive wording deterministically):
    # - cluster with the highest entropy AND density ratio:
    #   pick max mean_orientation_entropy, then among those pick max mean_density_ratio.
    max_entropy = float(profile["mean_orientation_entropy"].max())
    candidates_informal = profile[profile["mean_orientation_entropy"] == max_entropy]
    informal_idx = candidates_informal["mean_density_ratio"].idxmax()

    # Stable/formal selection (matches directive wording deterministically):
    # - cluster with the lowest entropy AND largest area:
    #   pick min mean_orientation_entropy, then among those pick max mean_area_m2.
    min_entropy = float(profile["mean_orientation_entropy"].min())
    candidates_stable = profile[profile["mean_orientation_entropy"] == min_entropy]
    stable_idx = candidates_stable["mean_area_m2"].idxmax()

    profile["category"] = "Planned Residential"
    profile.loc[informal_idx, "category"] = "Informal/High Risk"
    profile.loc[stable_idx, "category"] = "Stable/Formal Infrastructure"

    # Map to numeric risk:
    # - Informal/High Risk => 8
    # - Stable/Formal Infrastructure => 1
    # - Planned Residential => 4
    profile["risk_score"] = profile["category"].map(
        {
            "Informal/High Risk": 8,
            "Upgrading Zone": 4,  # kept for compatibility with older naming
            "Planned Residential": 4,
            "Stable/Formal Infrastructure": 1,
        }
    )
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Academic KMeans clustering + ARI stability ensemble.")
    parser.add_argument(
        "--csv",
        default="kanombe_features_academic.csv",
        help="Input features CSV path (default: kanombe_features_academic.csv in repo root).",
    )
    parser.add_argument(
        "--output_csv",
        default="kanombe_clustered_final.csv",
        help="Output clustered CSV path (default: kanombe_clustered_final.csv in repo root).",
    )
    parser.add_argument(
        "--paper_dir",
        default=str(Path("paper") / "figures"),
        help="Output directory for figures (default: paper/figures).",
    )
    args = parser.parse_args()

    # Keep a figure size default (font sizing handled globally above)
    plt.rcParams.update({"figure.figsize": (10, 6)})

    repo_root = Path(__file__).resolve().parents[1]
    features_csv = repo_root / args.csv
    if not features_csv.exists():
        raise FileNotFoundError(f"Could not find input CSV: {features_csv}")

    df = pd.read_csv(features_csv)

    # Drop NaNs as required by the directive.
    df = df.dropna(axis=0).reset_index(drop=True)

    # Extract numeric feature columns for clustering:
    # - exclude any obvious ID/string columns by selecting numeric dtype only.
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    if numeric_df.shape[1] < 2:
        raise ValueError("Not enough numeric columns available for clustering after filtering NaNs.")

    x_raw = numeric_df.values

    scaler = RobustScaler()
    x_scaled = scaler.fit_transform(x_raw)

    # --- K sweep validation (K=2..5, 50 seeds each) ---
    k_values = [2, 3, 4, 5]
    n_seeds = 50

    inertia_means = []
    sil_means = []
    ch_means = []
    db_means = []

    for k in k_values:
        inertias = []
        sils = []
        chs = []
        dbs = []

        for seed in range(n_seeds):
            inertia, labels = _kmeans_fit_for_metrics(x_scaled, k, seed)
            inertias.append(inertia)

            # Silhouette requires at least 2 clusters and labels variety.
            if len(np.unique(labels)) > 1 and k > 1:
                # Silhouette can be expensive for large n; use deterministic subsampling
                # for fast, stable K-sweep visualization.
                sample_size = int(min(2000, x_scaled.shape[0]))
                sils.append(float(silhouette_score(x_scaled, labels, sample_size=sample_size, random_state=42)))
            else:
                sils.append(np.nan)

            chs.append(float(calinski_harabasz_score(x_scaled, labels)))
            dbs.append(float(davies_bouldin_score(x_scaled, labels)))

        inertia_means.append(float(np.nanmean(inertias)))
        sil_means.append(float(np.nanmean(sils)))
        ch_means.append(float(np.nanmean(chs)))
        db_means.append(float(np.nanmean(dbs)))

    # Save elbow plot: Inertia vs K
    paper_dir = Path(repo_root) / args.paper_dir
    paper_dir.mkdir(parents=True, exist_ok=True)

    fig1_path = paper_dir / "fig1_elbow_plot.png"
    plt.figure()
    plt.plot(k_values, inertia_means, marker="o")
    plt.xlabel("K")
    plt.ylabel("Mean Inertia")
    plt.title("Elbow plot: Inertia vs K")
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    # Save validation indices plot: Silhouette, CH, DB vs K
    fig2_path = paper_dir / "fig2_validation_indices.png"
    plt.figure()
    plt.plot(k_values, sil_means, marker="o", label="Silhouette")
    plt.plot(k_values, ch_means, marker="o", label="Calinski-Harabasz")
    plt.plot(k_values, db_means, marker="o", label="Davies-Bouldin")
    plt.xlabel("K")
    plt.title("Validation indices vs K (mean across seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    # --- K=3 stability ensemble with ARI ---
    k_consensus = 3
    labels_ensemble = []

    for seed in range(n_seeds):
        labels_seed = _kmeans_labels(x_scaled, k_consensus, seed)
        labels_ensemble.append(labels_seed)

    labels_ensemble = np.asarray(labels_ensemble)  # (n_seeds, n_buildings)
    n_runs = labels_ensemble.shape[0]

    # Pairwise ARI matrix
    ari = np.eye(n_runs, dtype=float)
    pairwise_vals = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            v = float(adjusted_rand_score(labels_ensemble[i], labels_ensemble[j]))
            ari[i, j] = v
            ari[j, i] = v
            pairwise_vals.append(v)

    pairwise_vals = np.asarray(pairwise_vals, dtype=float)
    mean_pairwise_ari = float(pairwise_vals.mean())
    min_ari = float(pairwise_vals.min())
    max_ari = float(pairwise_vals.max())

    print("K=3 stability ensemble (pairwise ARI):")
    print(f"Mean Pairwise ARI: {mean_pairwise_ari}")
    print(f"Min ARI: {min_ari}")
    print(f"Max ARI: {max_ari}")

    # Choose consensus run: highest average ARI against all others
    mean_ari_for_run = ari.mean(axis=1)
    consensus_idx = int(mean_ari_for_run.argmax())
    labels_consensus = labels_ensemble[consensus_idx]

    print(f"Consensus Run Index (highest mean ARI): {consensus_idx}")

    df = df.copy()
    df["cluster_id"] = labels_consensus.astype(int)

    # --- Interpretive labeling via morphological profiling ---
    profile = compute_cluster_profile(df)
    df = df.merge(profile[["cluster_id", "category", "risk_score"]], on="cluster_id", how="left")

    # Console report table
    profile_print = profile[
        ["cluster_id", "mean_area_m2", "mean_density_ratio", "mean_orientation_entropy", "category", "risk_score"]
    ].copy()
    print("\nCluster Profile Table (interpretive morphological profiling):")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(profile_print.to_string(index=False))

    # --- Figures ---
    # fig3 PCA scatter of K=3 consensus clusters
    # Use X_scaled, reduced to 2D via PCA.
    pca = PCA(n_components=2, random_state=0)
    x_pca = pca.fit_transform(x_scaled)

    fig3_path = paper_dir / "fig3_pca_cluster_scatter.png"
    plt.figure()
    # High-contrast palette and larger markers for 2-column legibility
    palette = sns.color_palette("Set1", n_colors=len(np.unique(df["cluster_id"].values)))
    colors = [palette[int(c) % len(palette)] for c in df["cluster_id"].values]
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=colors, s=45, alpha=0.85, linewidths=0.0)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("K=3 PCA embedding (consensus clusters)")
    plt.tight_layout()
    plt.savefig(fig3_path, dpi=300)
    plt.close()

    # fig4 star metric boxplots: orientation_entropy and density_ratio by cluster
    fig4_path = paper_dir / "fig4_star_metric_boxplots.png"
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1, 2, 1)
    sns.boxplot(x="cluster_id", y="orientation_entropy", data=df, ax=ax1, linewidth=2.0)
    ax1.set_title("orientation_entropy by cluster")
    ax1.set_xlabel("cluster_id")
    ax1.set_ylabel("orientation_entropy")

    ax2 = plt.subplot(1, 2, 2)
    sns.boxplot(x="cluster_id", y="density_ratio", data=df, ax=ax2, linewidth=2.0)
    ax2.set_title("density_ratio by cluster")
    ax2.set_xlabel("cluster_id")
    ax2.set_ylabel("density_ratio")

    plt.tight_layout()
    plt.savefig(fig4_path, dpi=300)
    plt.close()

    # fig1/fig2 already created; requirement asks for exactly 4 publication-ready figures.

    # --- Pairwise ARI matrix heatmap required by earlier directives? ---
    # This stage's directive requested 4 specific names:
    #   fig1_elbow_plot.png
    #   fig2_validation_indices.png
    #   fig3_pca_cluster_scatter.png
    #   fig4_star_metric_boxplots.png
    # We comply exactly with those 4 outputs.

    # --- Final output CSV ---
    output_csv = repo_root / args.output_csv
    df.to_csv(output_csv, index=False)
    print(f"\nSaved clustered output CSV: {output_csv}")


if __name__ == "__main__":
    main()

