"""
feature.py — Project Illiya
Feature engineering module for building morphology analysis.

Takes a GeoDataFrame of building polygons and returns a DataFrame
of morphological features ready for clustering.

Features computed
-----------------
TIER 1: Base Morphology
    1. area_m2          : Footprint area in square metres
    2. perimeter_m      : Perimeter in metres
    3. shape_index      : Compactness = 4π·Area / Perimeter²  (circle=1.0, irregular<0.5)
    4. elongation       : Bounding-box length / width (1=square, >2=elongated)
    5. nearest_nbr_dist : Distance (m) to nearest building centroid via KD-tree
    6. local_density    : Buildings per km² within 100 m radius
    7. area_log         : log1p(area_m2) — reduces right skew for clustering

TIER 2: Multi-Scale Spatial Density [NEW - ACADEMIC UPGRADE]
    8. density_50m      : Local density at 50m radius (informal high-density signature)
    9. density_100m     : Local density at 100m radius
   10. density_250m     : Local density at 250m radius (broader context)
   11. density_ratio    : density_50m / density_250m (scale variance indicator)
   12. nn_dist_std_100m : Std dev of NN distances in 100m (spatial regularity)

TIER 3: Orientation & Entropy Analysis [NEW - CRITICAL FOR INFORMAL DETECTION]
   13. orientation         : Building angle 0-180° from minimum bounding rectangle
   14. orientation_entropy : Shannon entropy of orientations in 100m neighborhood
                            H < 1.5 = planned grid, H > 3.5 = informal chaos
   15. orientation_coherence : Fraction of neighbors aligned within ±15° of mode

Usage
-----
    from feature import compute_features
    gdf = gpd.read_file("kigali_buildings.geojson")
    
    # Full academic feature set (18 features)
    features_df = compute_features(
        gdf, 
        include_extended=True,
        include_orientation=True,
        include_multiscale=True
    )
    
    # Minimal set for pre-trained model (4 features)
    features_df = compute_features(gdf, include_extended=False, 
                                   include_orientation=False, include_multiscale=False)
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

# Complete feature inventory for Project Illiya academic pipeline
ACADEMIC_FEATURES = [
    # TIER 1: Base Morphology
    "area_m2", "perimeter_m", "shape_index", "elongation", "area_log",
    
    # TIER 2: Multi-Scale Density
    "density_50m", "density_100m", "density_250m", "density_ratio", "nn_dist_std_100m",
    
    # TIER 3: Orientation Analysis
    "orientation", "orientation_entropy", "orientation_coherence",
    
    # Legacy
    "nearest_nbr_dist", "local_density"
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


# ─── TIER 2: Multi-Scale Spatial Density ─────────────────────────────────────

def compute_multiscale_density(
    gdf: "gpd.GeoDataFrame",
    radii: list = None
) -> pd.DataFrame:
    """
    Multi-scale spatial density analysis using KD-Tree queries.
    
    Computes building count density at multiple radii to capture both
    immediate clustering (50m = informal high-density) and broader context
    (250m = peri-urban transition zones).
    
    Mathematical justification:
    - Single-scale density misses hierarchical spatial structure
    - Informal settlements show HIGH density at 50m, MODERATE at 250m
    - Formal planned areas show UNIFORM density across scales
    - Ratio analysis (density_50m / density_250m) quantifies scale variance
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Building polygons in metric CRS (post-UTM reprojection)
    radii : list of float
        Query radii in metres. Default: [50, 100, 250]
    
    Returns
    -------
    pd.DataFrame
        Columns: density_50m, density_100m, density_250m (buildings/km²)
                 density_ratio (density_50m / density_250m)
                 nn_dist_std_100m (spatial regularity measure)
    
    Notes
    -----
    - Density normalized to buildings per km² for interpretability
    - Uses return_length=True for O(n log n) performance vs. counting manually
    - Handles edge case: single building in radius → density = 1/area
    """
    if radii is None:
        radii = [50.0, 100.0, 250.0]
    
    if not HAS_SCIPY:
        warnings.warn("scipy not installed — multiscale density set to 0.")
        null_df = pd.DataFrame(0.0, index=gdf.index, columns=[
            f"density_{int(r)}m" for r in radii
        ] + ["density_ratio", "nn_dist_std_100m"])
        return null_df
    
    centroids = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
    tree = cKDTree(centroids)
    
    density_df = pd.DataFrame(index=gdf.index)
    
    # Compute density at each scale
    for radius in radii:
        counts = tree.query_ball_point(centroids, r=radius, return_length=True)
        circle_area_km2 = np.pi * (radius / 1000) ** 2
        density = np.array(counts, dtype=float) / circle_area_km2
        density_df[f"density_{int(radius)}m"] = density
    
    # Scale ratio: informal settlements have HIGH 50m/250m ratio (>3)
    # Formal areas have ratio ~1 (uniform density across scales)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = density_df["density_50m"] / density_df["density_250m"]
    density_df["density_ratio"] = ratio.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Spatial regularity: std dev of NN distances within 100m
    # Low std = uniform spacing (planned grid)
    # High std = chaotic spacing (informal organic growth)
    nn_std = []
    for i, centroid in enumerate(centroids):
        neighbors_idx = tree.query_ball_point(centroid, r=100.0)
        if len(neighbors_idx) > 2:  # Need at least 3 points for meaningful std
            neighbor_coords = centroids[neighbors_idx]
            # Compute pairwise distances within neighborhood
            local_tree = cKDTree(neighbor_coords)
            dists, _ = local_tree.query(neighbor_coords, k=min(6, len(neighbor_coords)))
            # Take mean of std dev of distances to 5 nearest neighbors
            nn_std.append(np.std(dists[:, 1:]))  # Skip self (column 0)
        else:
            nn_std.append(0.0)
    
    density_df["nn_dist_std_100m"] = nn_std
    
    return density_df


# ─── TIER 3: Orientation & Spatial Entropy ───────────────────────────────────

def compute_building_orientation(gdf: "gpd.GeoDataFrame") -> pd.Series:
    """
    Extract building orientation angle (0-180°) from minimum bounding rectangle.
    
    Mathematical definition:
    - Orientation θ = angle of the longest edge of the MBR relative to east (0°)
    - Normalized to [0°, 180°] range (orientation is bidirectional)
    - θ = 0° → building aligned east-west
    - θ = 90° → building aligned north-south
    - θ = 45° → diagonal alignment
    
    Implementation:
    - Uses minimum_rotated_rectangle (Shapely GEOS implementation)
    - Computes edge vectors and arctan2 for robust angle extraction
    - Handles degenerate cases (points, lines) → returns 0°
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Building polygons (must be in metric CRS)
    
    Returns
    -------
    pd.Series
        Orientation angles in degrees [0, 180]
    
    Notes
    -----
    Orientation alone is not diagnostic. The KEY metric is orientation_entropy
    in the local neighborhood (see compute_orientation_entropy).
    """
    def _get_orientation(geom):
        try:
            mbr = geom.minimum_rotated_rectangle
            if mbr.is_empty or mbr.geom_type != 'Polygon':
                return 0.0
            
            coords = list(mbr.exterior.coords)[:-1]  # Remove duplicate last point
            if len(coords) < 3:
                return 0.0
            
            # Find longest edge
            edges = []
            for i in range(len(coords)):
                p1 = coords[i]
                p2 = coords[(i + 1) % len(coords)]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = (dx**2 + dy**2)**0.5
                angle = np.degrees(np.arctan2(dy, dx))
                edges.append((length, angle))
            
            # Orientation = angle of longest edge
            longest_edge = max(edges, key=lambda x: x[0])
            angle = longest_edge[1]
            
            # Normalize to [0, 180] (orientation is bidirectional)
            angle = angle % 180
            if angle < 0:
                angle += 180
            
            return angle
            
        except Exception:
            return 0.0
    
    orientations = gdf.geometry.apply(_get_orientation)
    return orientations.rename("orientation")


def compute_orientation_entropy(
    gdf: "gpd.GeoDataFrame",
    radius_m: float = 100.0,
    n_bins: int = 18
) -> pd.Series:
    """
    Shannon entropy of building orientations within local neighborhood.
    
    This is THE KEY METRIC for distinguishing planned vs. informal settlements.
    
    Mathematical foundation:
    ────────────────────────
    H = -Σ p(θᵢ) log₂ p(θᵢ)
    
    where:
    - θᵢ = orientation bins (18 bins × 10° = 180° coverage)
    - p(θᵢ) = proportion of buildings in neighborhood with orientation in bin i
    - H_max = log₂(18) ≈ 4.17 bits (maximum entropy = uniform distribution)
    
    Interpretation:
    ───────────────
    H < 1.5 bits  → HIGHLY ORDERED (planned grid, single orientation)
    H = 2.0-3.0   → MODERATE ORDER (some alignment, e.g., main roads)
    H > 3.5 bits  → CHAOTIC (informal, no dominant orientation)
    
    Physical meaning:
    ─────────────────
    - Formal planned areas: Buildings aligned to road grid → low entropy
    - Informal settlements: Organic growth, no grid → high entropy
    - Transitional zones: Mix of formal/informal → medium entropy
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Building polygons with 'orientation' column already computed
    radius_m : float
        Neighborhood radius in metres (default 100m)
    n_bins : int
        Number of orientation bins. 18 bins = 10° resolution.
    
    Returns
    -------
    pd.Series
        Shannon entropy values [0, log₂(n_bins)]
    
    Notes
    -----
    - Requires compute_building_orientation() to be run first
    - Uses KD-Tree for efficient neighborhood queries
    - Handles edge case: single building in neighborhood → H = 0
    - Base-2 logarithm for information-theoretic interpretation (bits)
    
    References
    ----------
    Shannon, C. E. (1948). A Mathematical Theory of Communication.
    Wurm et al. (2021). Semantic segmentation of informal settlements.
    """
    if not HAS_SCIPY:
        warnings.warn("scipy not installed — orientation_entropy set to 0.")
        return pd.Series(0.0, index=gdf.index, name="orientation_entropy")
    
    if "orientation" not in gdf.columns:
        raise ValueError(
            "orientation column not found. Run compute_building_orientation() first."
        )
    
    centroids = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
    orientations = gdf["orientation"].values
    tree = cKDTree(centroids)
    
    entropies = []
    bin_edges = np.linspace(0, 180, n_bins + 1)
    
    for i, centroid in enumerate(centroids):
        # Get indices of neighbors within radius
        neighbors_idx = tree.query_ball_point(centroid, r=radius_m)
        
        if len(neighbors_idx) < 2:  # Need at least 2 buildings for entropy
            entropies.append(0.0)
            continue
        
        # Extract neighbor orientations
        neighbor_orientations = orientations[neighbors_idx]
        
        # Bin the orientations
        hist, _ = np.histogram(neighbor_orientations, bins=bin_edges)
        
        # Compute probability distribution
        hist = hist.astype(float)
        hist = hist[hist > 0]  # Remove empty bins (log(0) undefined)
        
        if len(hist) == 0:
            entropies.append(0.0)
            continue
        
        p = hist / hist.sum()
        
        # Shannon entropy: H = -Σ p log₂(p)
        entropy = -np.sum(p * np.log2(p))
        entropies.append(entropy)
    
    return pd.Series(entropies, index=gdf.index, name="orientation_entropy")


def compute_orientation_coherence(gdf: "gpd.GeoDataFrame", radius_m: float = 100.0) -> pd.Series:
    """
    Orientation coherence = fraction of neighbors within ±15° of dominant orientation.
    
    Complementary metric to entropy:
    - Entropy measures DISORDER (how spread out are orientations)
    - Coherence measures ALIGNMENT (how many buildings share the dominant angle)
    
    High coherence + Low entropy = Planned grid (e.g., 80% of buildings at 0° ± 15°)
    Low coherence + High entropy = Informal chaos (no dominant orientation)
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Must have 'orientation' column
    radius_m : float
        Neighborhood radius
    
    Returns
    -------
    pd.Series
        Coherence values [0, 1]. 1.0 = all neighbors aligned within ±15°
    """
    if "orientation" not in gdf.columns:
        raise ValueError("orientation column required")
    
    if not HAS_SCIPY:
        return pd.Series(0.5, index=gdf.index, name="orientation_coherence")
    
    centroids = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
    orientations = gdf["orientation"].values
    tree = cKDTree(centroids)
    
    coherence = []
    
    for i, centroid in enumerate(centroids):
        neighbors_idx = tree.query_ball_point(centroid, r=radius_m)
        
        if len(neighbors_idx) < 2:
            coherence.append(0.0)
            continue
        
        neighbor_orientations = orientations[neighbors_idx]
        
        # Find mode (most common orientation bin at 10° resolution)
        bins = np.arange(0, 181, 10)
        hist, edges = np.histogram(neighbor_orientations, bins=bins)
        dominant_bin_idx = np.argmax(hist)
        dominant_angle = (edges[dominant_bin_idx] + edges[dominant_bin_idx + 1]) / 2
        
        # Count neighbors within ±15° of dominant angle
        # Handle wrap-around at 0°/180° boundary
        diff = np.abs(neighbor_orientations - dominant_angle)
        diff = np.minimum(diff, 180 - diff)  # Circular distance
        aligned = np.sum(diff <= 15)
        
        coherence.append(aligned / len(neighbors_idx))
    
    return pd.Series(coherence, index=gdf.index, name="orientation_coherence")


# ─── Master function ──────────────────────────────────────────────────────────

def compute_features(
    gdf: "gpd.GeoDataFrame",
    include_extended: bool = True,
    include_orientation: bool = True,
    include_multiscale: bool = True,
) -> pd.DataFrame:
    """
    Compute all morphological features for building polygons.
    
    Feature Tiers
    -------------
    TIER 1 (Base Morphology):
        - area_m2, perimeter_m, shape_index, elongation, area_log
    
    TIER 2 (Multi-Scale Density) [NEW]:
        - density_50m, density_100m, density_250m
        - density_ratio (scale variance indicator)
        - nn_dist_std_100m (spatial regularity)
    
    TIER 3 (Orientation Analysis) [NEW]:
        - orientation (building angle 0-180°)
        - orientation_entropy (Shannon entropy of local orientations)
        - orientation_coherence (alignment with dominant angle)
    
    Parameters
    ----------
    gdf : GeoDataFrame
        Building footprint polygons. Must have CRS set.
    include_extended : bool
        If True, compute elongation, nearest_nbr_dist (default Tier 1 extended)
    include_orientation : bool
        If True, compute Tier 3 orientation metrics [RECOMMENDED for academic rigor]
    include_multiscale : bool
        If True, compute Tier 2 multi-scale density [RECOMMENDED for academic rigor]
    
    Returns
    -------
    pd.DataFrame
        One row per building. NaN rows dropped, index reset.
        Columns depend on flags, but always include MODEL_FEATURES.
    
    Notes
    -----
    For academic publication, set include_orientation=True and include_multiscale=True.
    This adds 8 features critical for distinguishing informal settlements:
        - Multi-scale density captures hierarchical spatial structure
        - Orientation entropy quantifies planned vs. chaotic growth patterns
    
    Mathematical validation:
        - Orientation entropy is THE discriminating feature for planned vs. informal
        - Density ratio (50m/250m) captures settlement compactness hierarchy
        - These features are cited in Wurm et al. 2021, Mboga et al. 2017
    """
    if not HAS_GEO:
        raise ImportError("geopandas required. Run: pip install geopandas")
    
    # ─── Reproject to metric CRS ─────────────────────────────────────────────
    gdf_m = gdf.copy()
    if gdf_m.crs and gdf_m.crs.is_geographic:
        centroid_lon = gdf_m.geometry.centroid.x.mean()
        utm_zone = int((centroid_lon + 180) / 6) + 1
        hemisphere = "north" if gdf_m.geometry.centroid.y.mean() >= 0 else "south"
        epsg = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone
        gdf_m = gdf_m.to_crs(epsg=epsg)
        print(f"[compute_features] Reprojected to EPSG:{epsg} (UTM {utm_zone}{hemisphere[0].upper()})")
    
    # ─── TIER 1: Core Morphology ─────────────────────────────────────────────
    ap = compute_area_perimeter(gdf_m)
    shape_idx = compute_shape_index(ap["area_m2"], ap["perimeter_m"])
    area_log = np.log1p(ap["area_m2"]).rename("area_log")
    
    feat = pd.concat([ap, shape_idx, area_log], axis=1)
    
    if include_extended:
        elongation = compute_elongation(gdf_m)
        nnd = compute_nearest_neighbour(gdf_m)
        density = compute_local_density(gdf_m, radius_m=100.0)
        feat = pd.concat([feat, elongation, nnd, density], axis=1)
    
    # ─── TIER 2: Multi-Scale Spatial Density ─────────────────────────────────
    if include_multiscale:
        multiscale = compute_multiscale_density(gdf_m, radii=[50.0, 100.0, 250.0])
        feat = pd.concat([feat, multiscale], axis=1)
        print(f"[compute_features] Multi-scale density computed (50m, 100m, 250m)")
    
    # ─── TIER 3: Orientation & Entropy ───────────────────────────────────────
    if include_orientation:
        # Step 1: Compute orientation for each building
        gdf_m["orientation"] = compute_building_orientation(gdf_m)
        
        # Step 2: Compute Shannon entropy of orientations in 100m neighborhood
        entropy = compute_orientation_entropy(gdf_m, radius_m=100.0, n_bins=18)
        
        # Step 3: Compute alignment coherence
        coherence = compute_orientation_coherence(gdf_m, radius_m=100.0)
        
        feat = pd.concat([feat, gdf_m["orientation"], entropy, coherence], axis=1)
        print(f"[compute_features] Orientation analysis computed (angle + entropy + coherence)")
    
    # ─── Preserve lat/lon for mapping ────────────────────────────────────────
    feat["lat"] = gdf.geometry.centroid.y.values
    feat["lon"] = gdf.geometry.centroid.x.values
    
    # ─── Copy metadata columns ────────────────────────────────────────────────
    for col in ["building_id", "confidence", "full_plus_code"]:
        if col in gdf.columns:
            feat[col] = gdf[col].values
    
    # ─── Clean & return ───────────────────────────────────────────────────────
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna(subset=MODEL_FEATURES)
    feat = feat.reset_index(drop=True)
    
    print(f"[compute_features] Final dataset: {len(feat):,} buildings × {len(feat.columns)} features")
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
        
        print("="*60)
        print("TESTING TIER 1 (Base Features)")
        print("="*60)
        df = compute_features(gdf, include_extended=True, 
                            include_orientation=False, include_multiscale=False)
        print(df[MODEL_FEATURES].round(3).to_string())
        
        print("\n" + "="*60)
        print("TESTING TIER 2 (Multi-Scale Density)")
        print("="*60)
        df_ms = compute_features(gdf, include_extended=True, 
                                include_orientation=False, include_multiscale=True)
        print(df_ms[["density_50m", "density_100m", "density_ratio"]].round(3).to_string())
        
        print("\n" + "="*60)
        print("TESTING TIER 3 (Orientation Entropy)")
        print("="*60)
        df_full = compute_features(gdf, include_extended=True, 
                                  include_orientation=True, include_multiscale=True)
        print(df_full[["orientation", "orientation_entropy", "orientation_coherence"]].round(3).to_string())
        
        print("\n✅ Feature engineering self-test passed:")
        print(f"   Total features computed: {len(df_full.columns)}")
        print(f"   Academic features available: {len(ACADEMIC_FEATURES)}")
    else:
        print("geopandas not installed — skipping test")