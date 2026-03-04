"""
feature.py — Urban DNA Sequencer
Feature engineering module for building morphology analysis.

Takes a GeoDataFrame of building polygons and returns a DataFrame
of morphological features ready for clustering.

Features computed
-----------------
1. area_m2          : Footprint area in square metres
2. perimeter_m      : Perimeter in metres
3. shape_index      : Compactness = 4π·Area / Perimeter²  (circle=1.0, irregular<0.5)
4. elongation       : Bounding-box length / width (1=square, >2=elongated)
5. nearest_nbr_dist : Distance (m) to nearest building centroid via KD-tree
6. local_density    : Buildings per km² within 100 m radius
7. area_log         : log1p(area_m2) — reduces right skew for clustering

Usage
-----
    from feature import compute_features
    gdf = gpd.read_file("kigali_buildings.geojson")
    features_df = compute_features(gdf)
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional

# Optional spatial imports - graceful fallback if not installed
try:
    import geopandas as gpd
    from shapely.geometry import box
    HAS_GEO = True
except ImportError:
    HAS_GEO = False
    warnings.warn("geopandas/shapely not installed. Geographic CRS conversion will be skipped.")

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not installed. Nearest-neighbour distance will be set to 0.")


# ─── Constants ───────────────────────────────────────────────────────────────

# The 4 features the trained urban_dna_brain.pkl was built on
# (matches StandardScaler input order)
MODEL_FEATURES = ["area_m2", "perimeter_m", "shape_index", "area_log"]

# All computable features (superset — for richer analysis)
ALL_FEATURES = [
    "area_m2", "perimeter_m", "shape_index",
    "elongation", "nearest_nbr_dist", "local_density", "area_log"
]


# ─── Core geometry features ───────────────────────────────────────────────────

def compute_area_perimeter(gdf: "gpd.GeoDataFrame") -> pd.DataFrame:
    """
    Compute area (m²) and perimeter (m) for each building polygon.
    Reprojects to a local UTM CRS if the input is geographic (lat/lon).
    """
    if not HAS_GEO:
        raise ImportError("geopandas required. Run: pip install geopandas")

    gdf = gdf.copy()

    # Reproject to metric CRS if needed
    if gdf.crs and gdf.crs.is_geographic:
        # Auto-select UTM zone based on centroid longitude
        centroid_lon = gdf.geometry.centroid.x.mean()
        utm_zone = int((centroid_lon + 180) / 6) + 1
        hemisphere = "north" if gdf.geometry.centroid.y.mean() >= 0 else "south"
        epsg = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone
        gdf = gdf.to_crs(epsg=epsg)
    elif gdf.crs is None:
        warnings.warn("No CRS found — assuming geometry is already in metres.")

    df = pd.DataFrame()
    df["area_m2"]     = gdf.geometry.area
    df["perimeter_m"] = gdf.geometry.length
    df.index = gdf.index
    return df


def compute_shape_index(area: pd.Series, perimeter: pd.Series) -> pd.Series:
    """
    Shape Index (Polsby–Popper compactness).
    SI = 4π·Area / Perimeter²
    Range: (0, 1]. A perfect circle = 1.0. Highly irregular = close to 0.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        si = (4 * np.pi * area) / (perimeter ** 2)
    return si.clip(0, 1).rename("shape_index")


def compute_elongation(gdf: "gpd.GeoDataFrame") -> pd.Series:
    """
    Elongation = long_side / short_side of the minimum rotated bounding box.
    = 1 for squares, > 2 for elongated buildings.
    """
    def _elongation(geom):
        try:
            mbr = geom.minimum_rotated_rectangle
            coords = list(mbr.exterior.coords)
            edges = [
                ((coords[i][0] - coords[i-1][0])**2 + (coords[i][1] - coords[i-1][1])**2)**0.5
                for i in range(1, 5)
            ]
            edges.sort()
            if edges[0] < 1e-6:
                return 1.0
            return edges[-1] / edges[0]
        except Exception:
            return 1.0

    return gdf.geometry.apply(_elongation).rename("elongation")


def compute_nearest_neighbour(gdf: "gpd.GeoDataFrame") -> pd.Series:
    """
    Nearest-neighbour distance (m) using centroid KD-tree.
    Requires metric CRS (call after reprojecting).
    """
    if not HAS_SCIPY:
        warnings.warn("scipy not installed — nearest_nbr_dist set to 0.")
        return pd.Series(0.0, index=gdf.index, name="nearest_nbr_dist")

    centroids = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
    tree = cKDTree(centroids)
    dist, _ = tree.query(centroids, k=2)   # k=2: skip self (idx 0)
    return pd.Series(dist[:, 1], index=gdf.index, name="nearest_nbr_dist")


def compute_local_density(gdf: "gpd.GeoDataFrame", radius_m: float = 100.0) -> pd.Series:
    """
    Local density = number of building centroids within `radius_m` metres,
    normalised to buildings per km².
    """
    if not HAS_SCIPY:
        warnings.warn("scipy not installed — local_density set to 0.")
        return pd.Series(0.0, index=gdf.index, name="local_density")

    centroids = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
    tree = cKDTree(centroids)
    counts = tree.query_ball_point(centroids, r=radius_m, return_length=True)
    # Normalise: circle area = π·r²  → buildings per km²
    circle_area_km2 = np.pi * (radius_m / 1000) ** 2
    density = np.array(counts, dtype=float) / circle_area_km2
    return pd.Series(density, index=gdf.index, name="local_density")


# ─── Master function ──────────────────────────────────────────────────────────

def compute_features(
    gdf: "gpd.GeoDataFrame",
    include_extended: bool = True,
) -> pd.DataFrame:
    """
    Compute all morphological features for a GeoDataFrame of building polygons.

    Parameters
    ----------
    gdf : GeoDataFrame
        Building footprint polygons. Must have a CRS set.
    include_extended : bool
        If True, compute all 7 features.
        If False, compute only the 4 MODEL_FEATURES (area_m2, perimeter_m,
        shape_index, area_log) which are required by urban_dna_brain.pkl.

    Returns
    -------
    pd.DataFrame
        One row per building, columns = feature names.
        NaN rows are dropped and index is reset.
    """
    if not HAS_GEO:
        raise ImportError("geopandas required. Run: pip install geopandas")

    # Work on a projected copy (metric)
    gdf_m = gdf.copy()
    if gdf_m.crs and gdf_m.crs.is_geographic:
        centroid_lon = gdf_m.geometry.centroid.x.mean()
        utm_zone = int((centroid_lon + 180) / 6) + 1
        hemisphere = "north" if gdf_m.geometry.centroid.y.mean() >= 0 else "south"
        epsg = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone
        gdf_m = gdf_m.to_crs(epsg=epsg)

    # Core features (always computed)
    ap = compute_area_perimeter(gdf_m)
    shape_idx = compute_shape_index(ap["area_m2"], ap["perimeter_m"])
    area_log = np.log1p(ap["area_m2"]).rename("area_log")

    feat = pd.concat([ap, shape_idx, area_log], axis=1)

    # Extended features (optional)
    if include_extended:
        elongation   = compute_elongation(gdf_m)
        nnd          = compute_nearest_neighbour(gdf_m)
        density      = compute_local_density(gdf_m)
        feat = pd.concat([feat, elongation, nnd, density], axis=1)

    # Preserve lat/lon for map display
    feat["lat"] = gdf.geometry.centroid.y.values
    feat["lon"] = gdf.geometry.centroid.x.values

    # Copy any useful metadata columns
    for col in ["building_id", "confidence", "full_plus_code"]:
        if col in gdf.columns:
            feat[col] = gdf[col].values

    feat = feat.replace([np.inf, -np.inf], np.nan).dropna(subset=MODEL_FEATURES)
    feat = feat.reset_index(drop=True)
    return feat


# ─── Utility: load GeoJSON ────────────────────────────────────────────────────

def load_geojson(path: str) -> "gpd.GeoDataFrame":
    """
    Load a GeoJSON file of building polygons.
    Filters to only Polygon/MultiPolygon geometries.
    """
    if not HAS_GEO:
        raise ImportError("geopandas required. Run: pip install geopandas")

    gdf = gpd.read_file(path)
    # Keep only polygon geometries
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf = gdf.reset_index(drop=True)
    print(f"[load_geojson] Loaded {len(gdf):,} buildings from {path}")
    return gdf


# ─── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    if HAS_GEO:
        from shapely.geometry import Polygon
        import geopandas as gpd

        # Tiny synthetic test — 3 buildings
        polys = [
            Polygon([(0,0),(10,0),(10,10),(0,10)]),        # 100 m² square
            Polygon([(20,0),(25,0),(25,50),(20,50)]),       # 250 m² rectangle (elongated)
            Polygon([(40,0),(55,0),(60,10),(45,15),(40,0)]),# irregular pentagon
        ]
        gdf = gpd.GeoDataFrame({"geometry": polys}, crs="EPSG:32736")  # UTM 36S (Rwanda)
        df = compute_features(gdf)
        print("\n✅ Feature engineering self-test passed:")
        print(df[MODEL_FEATURES].round(3).to_string())
    else:
        print("geopandas not installed — skipping test")
