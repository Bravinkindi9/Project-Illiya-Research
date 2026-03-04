"""
pipeline.py — Urban DNA Sequencer
End-to-end reproducible pipeline: GeoJSON → features → model → results CSV/GeoJSON.

Usage
-----
    # Basic (uses existing model)
    python pipeline.py --input data/kigali_buildings.geojson

    # Specify output directory
    python pipeline.py --input data/kigali_buildings.geojson --output outputs/

    # Retrain model from scratch
    python pipeline.py --input data/kigali_buildings.geojson --retrain --k 3

    # Quick smoke-test with synthetic data (no GeoJSON needed)
    python pipeline.py --test
"""

import argparse
import json
import os
import sys
import warnings
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Project imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature import compute_features, load_geojson, MODEL_FEATURES

# ─── Paths ──────────────────────────────────────────────────────────────────

ROOT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "apps", "urban_dna_brain.pkl")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")


# ─── Cluster labelling ────────────────────────────────────────────────────────

CLUSTER_LABELS = {
    0: "Informal / High Risk",
    1: "Upgrading Zone",
    2: "Stable / Formal",
}

RISK_SCORE = {
    "Informal / High Risk": 8,
    "Upgrading Zone":       4,
    "Stable / Formal":      1,
}


def label_clusters(km: KMeans, X_scaled: np.ndarray) -> dict:
    """
    Auto-label clusters by morphological profile.
    Informal = smallest area + highest density (lowest cluster center values for
    area features, since density is inverse).
    Returns {cluster_id: label}.
    """
    centers = km.cluster_centers_
    # Use area dimension (index 0) to rank: smallest area → most informal
    rank_by_area = np.argsort(centers[:, 0])  # ascending area
    labels = {}
    category_order = ["Informal / High Risk", "Upgrading Zone", "Stable / Formal"]
    for rank, cluster_id in enumerate(rank_by_area):
        labels[int(cluster_id)] = category_order[rank]
    return labels


# ─── Model loading / training ─────────────────────────────────────────────────

def load_model(path: str = MODEL_PATH) -> dict:
    """Load urban_dna_brain.pkl. Returns dict with kmeans_model, scaler, metadata."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Run with --retrain to build one.")
    brain = joblib.load(path)
    print(f"[pipeline] ✅ Loaded model from {path}")
    print(f"           K={brain['kmeans_model'].n_clusters}, "
          f"inertia={brain['metadata'].get('inertia', '?'):.0f}")
    return brain


def train_model(X_scaled: np.ndarray, k: int = 3, random_state: int = 42) -> KMeans:
    """Train K-Means on standardised features."""
    print(f"[pipeline] 🔧 Training KMeans k={k} on {len(X_scaled):,} buildings…")
    t0 = time.time()
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    km.fit(X_scaled)
    t1 = time.time()
    print(f"[pipeline]    Done in {t1-t0:.1f}s — inertia={km.inertia_:.1f}")
    return km


def elbow_analysis(X_scaled: np.ndarray, k_range: range = range(2, 9)) -> dict:
    """Run elbow analysis. Returns dict of {k: inertia}."""
    print("[pipeline] 📊 Running elbow analysis…")
    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(X_scaled)
        results[k] = km.inertia_
        sil = silhouette_score(X_scaled, km.labels_) if len(set(km.labels_)) > 1 else 0
        print(f"   k={k}: inertia={km.inertia_:.0f}, silhouette={sil:.3f}")
    return results


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_model(km: KMeans, X_scaled: np.ndarray, labels: np.ndarray) -> dict:
    """Compute clustering quality metrics."""
    metrics = {}
    if len(set(labels)) < 2:
        return metrics
    metrics["silhouette_score"]    = round(float(silhouette_score(X_scaled, labels)), 4)
    metrics["davies_bouldin_index"] = round(float(davies_bouldin_score(X_scaled, labels)), 4)
    metrics["inertia"]             = round(float(km.inertia_), 2)
    metrics["n_clusters"]          = int(km.n_clusters)

    print("\n[pipeline] 📈 Model Evaluation:")
    print(f"   Silhouette Score    : {metrics['silhouette_score']:>8.4f}  (best=1.0, >0.4 is good)")
    print(f"   Davies-Bouldin Index: {metrics['davies_bouldin_index']:>8.4f}  (best=0.0, lower=better)")
    print(f"   Inertia             : {metrics['inertia']:>10.1f}")
    return metrics


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(
    geojson_path: str,
    model_path: str = MODEL_PATH,
    output_dir: str = OUTPUT_DIR,
    retrain: bool = False,
    k: int = 3,
    save_geojson: bool = True,
    save_csv: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline: load → features → cluster → label → save.

    Returns the result DataFrame with columns:
    lat, lon, area_m2, perimeter_m, shape_index, area_log,
    cluster_id, category, risk_score, [+ extended features if available]
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load GeoJSON
    print(f"\n[pipeline] 📂 Loading buildings from {geojson_path}…")
    gdf = load_geojson(geojson_path)
    print(f"[pipeline]    {len(gdf):,} buildings loaded")

    # 2. Feature engineering
    print("[pipeline] 🔧 Computing morphological features…")
    t0 = time.time()
    feat_df = compute_features(gdf, include_extended=True)
    print(f"[pipeline]    Features computed in {time.time()-t0:.1f}s for {len(feat_df):,} buildings")

    # 3. Prepare feature matrix (only MODEL_FEATURES for the scaler/model)
    X_raw = feat_df[MODEL_FEATURES].values

    # 4. Load or train model
    if retrain or not os.path.exists(model_path):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        km = train_model(X_scaled, k=k)
        cluster_label_map = label_clusters(km, X_scaled)
        brain = {
            "kmeans_model": km,
            "scaler": scaler,
            "metadata": {
                "model_type": "KMeans",
                "n_clusters": k,
                "feature_cols": MODEL_FEATURES,
                "inertia": km.inertia_,
                "n_iter": km.n_iter_,
                "cluster_label_map": cluster_label_map,
            },
        }
        joblib.dump(brain, model_path, compress=3)
        print(f"[pipeline] 💾 Model saved to {model_path}")
    else:
        brain = load_model(model_path)
        scaler = brain["scaler"]
        km     = brain["kmeans_model"]
        X_scaled = scaler.transform(X_raw)
        cluster_label_map = brain["metadata"].get(
            "cluster_label_map", label_clusters(km, X_scaled)
        )

    # 5. Predict
    raw_cluster_ids = km.predict(X_scaled).astype(int)

    # 6. Evaluate
    metrics = evaluate_model(km, X_scaled, raw_cluster_ids)

    # 7. Map cluster IDs to labels (re-label to ensure consistent category names)
    feat_df["cluster_id"] = raw_cluster_ids
    feat_df["category"]   = feat_df["cluster_id"].map(cluster_label_map)
    feat_df["risk_score"] = feat_df["category"].map(RISK_SCORE)

    # 8. Cluster distribution
    print("\n[pipeline] 🗂️  Cluster Distribution:")
    for cid, label in sorted(cluster_label_map.items()):
        count = (feat_df["cluster_id"] == cid).sum()
        pct   = 100 * count / len(feat_df)
        print(f"   Cluster {cid} [{label:<22}]: {count:>6,} buildings ({pct:.1f}%)")

    # 9. Save outputs
    city_name = os.path.splitext(os.path.basename(geojson_path))[0]
    timestamp = time.strftime("%Y%m%d_%H%M")

    if save_csv:
        csv_path = os.path.join(output_dir, f"{city_name}_results_{timestamp}.csv")
        # Also always overwrite the canonical result (used by the frontend)
        canonical_csv = os.path.join(output_dir, f"{city_name}_results.csv")
        feat_df.to_csv(csv_path, index=False)
        feat_df.to_csv(canonical_csv, index=False)
        print(f"\n[pipeline] ✅ CSV saved → {canonical_csv}")

    if save_geojson:
        geojson_path_out = os.path.join(output_dir, f"{city_name}_results_{timestamp}.geojson")
        canonical_geojson = os.path.join(output_dir, f"{city_name}_results.geojson")
        # Re-attach geometry
        result_gdf = gdf.copy()
        result_gdf = result_gdf.reset_index(drop=True)
        for col in ["cluster_id", "category", "risk_score",
                    "area_m2", "perimeter_m", "shape_index", "area_log"]:
            if col in feat_df.columns:
                result_gdf[col] = feat_df[col].values
        result_gdf.to_file(geojson_path_out, driver="GeoJSON")
        result_gdf.to_file(canonical_geojson, driver="GeoJSON")
        print(f"[pipeline] ✅ GeoJSON saved → {canonical_geojson}")

    # 10. Save metrics
    metrics_path = os.path.join(output_dir, f"{city_name}_metrics_{timestamp}.json")
    metrics["cluster_distribution"] = {
        str(cid): {"label": lbl, "count": int((feat_df["cluster_id"] == cid).sum())}
        for cid, lbl in cluster_label_map.items()
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[pipeline] ✅ Metrics saved → {metrics_path}")

    print(f"\n[pipeline] 🧬 Pipeline complete. {len(feat_df):,} buildings sequenced.\n")
    return feat_df


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _smoke_test():
    """Synthetic test — no GeoJSON needed."""
    try:
        import geopandas as gpd
        from shapely.geometry import Polygon
    except ImportError:
        print("geopandas not installed — pip install geopandas shapely")
        sys.exit(1)

    print("[smoke test] Creating 200 synthetic buildings…")
    np.random.seed(42)
    polys = []
    for _ in range(200):
        cx, cy = np.random.uniform(0, 1000, 2)
        w = np.random.uniform(5, 40)
        h = np.random.uniform(5, 40)
        polys.append(Polygon([(cx,cy),(cx+w,cy),(cx+w,cy+h),(cx,cy+h)]))
    gdf = gpd.GeoDataFrame({"geometry": polys}, crs="EPSG:32736")
    feat = compute_features(gdf, include_extended=True)
    print(feat[MODEL_FEATURES].describe().round(2).to_string())
    print("\n[smoke test] ✅ feature.py working correctly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Urban DNA Sequencer — End-to-end pipeline"
    )
    parser.add_argument("--input",   "-i", help="Path to input GeoJSON file")
    parser.add_argument("--output",  "-o", default=OUTPUT_DIR, help="Output directory (default: outputs/)")
    parser.add_argument("--model",   "-m", default=MODEL_PATH,  help="Path to .pkl model")
    parser.add_argument("--retrain", "-r", action="store_true",  help="Retrain model from scratch")
    parser.add_argument("--k",             type=int, default=3,  help="Number of clusters if retraining")
    parser.add_argument("--elbow",         action="store_true",  help="Run elbow analysis (k=2..8)")
    parser.add_argument("--test",          action="store_true",  help="Smoke test with synthetic data")
    args = parser.parse_args()

    if args.test:
        _smoke_test()
    elif args.input is None:
        parser.print_help()
        print("\n💡 Quick test: python pipeline.py --test")
    else:
        if not os.path.exists(args.input):
            print(f"❌ Input file not found: {args.input}")
            sys.exit(1)
        run_pipeline(
            geojson_path=args.input,
            model_path=args.model,
            output_dir=args.output,
            retrain=args.retrain,
            k=args.k,
        )
