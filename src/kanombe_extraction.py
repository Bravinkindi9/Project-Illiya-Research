"""
kanombe_extraction.py — Project Illiya Data Acquisition
Google Earth Engine script to extract Open Buildings V3 for Kanombe, Kigali

Area of Interest: Kanombe, Kigali, Rwanda
- Mixed morphology: Formal (airport, planned residential) + Informal settlements
- Critical test case for orientation entropy & multi-scale density validation

Output: kanombe_buildings.geojson (local file, ready for feature.py)

Requirements:
    pip install earthengine-api geemap geopandas folium

Authentication:
    Run once: ee.Authenticate()
    Then this script will work automatically

Usage:
    python kanombe_extraction.py
"""

# Print immediately so user knows script started (geemap import can take 30-60 sec)
print("Starting Kanombe extraction... (loading libraries, please wait 30-60 sec)", flush=True)
import sys
sys.stdout.flush()

import ee
import geemap
import geopandas as gpd
from shapely.geometry import box
import json

# ─── Step 1: Initialize Earth Engine ─────────────────────────────────────────

print("="*70)
print("PROJECT ILLIYA - KANOMBE DATA EXTRACTION")
print("="*70)

try:
    # CRITICAL FIX: Always specify project, even if already authenticated
    ee.Initialize(project='gee-mymaap')
    print("[OK] Earth Engine initialized successfully (project: gee-mymaap)")
except Exception as e:
    print(f"[!] Earth Engine initialization failed (project 'gee-mymaap'): {e}")
    print("   Trying without project ID...")
    try:
        ee.Initialize()
        print("[OK] Earth Engine initialized (default project)")
    except Exception as e2:
        print(f"[X] Init without project also failed: {e2}")
        print("\nAttempting authentication (opens browser)...")
        try:
            ee.Authenticate()
            ee.Initialize()
            print("[OK] Earth Engine authenticated and initialized")
        except Exception as e3:
            print(f"[X] FATAL: Could not initialize Earth Engine: {e3}")
            print("\nTROUBLESHOOTING:")
            print("   1. Sign up at https://earthengine.google.com")
            print("   2. Run: python -c \"import ee; ee.Authenticate()\"")
            print("   3. Or set project ID on line 40: ee.Initialize(project='YOUR-PROJECT-ID')")
            sys.exit(1)


# ─── Step 2: Define Kanombe AOI ──────────────────────────────────────────────

# Kanombe bounding box (verified via Google Maps)
# Covers: Kigali International Airport + surrounding residential + informal settlements
# Coordinates: [min_lon, min_lat, max_lon, max_lat]

KANOMBE_BBOX = [
    30.1250,  # min_lon (west)
    -1.9650,  # min_lat (south)
    30.1450,  # max_lon (east)
    -1.9450   # max_lat (north)
]

print("\n" + "="*70)
print("AREA OF INTEREST: Kanombe, Kigali, Rwanda")
print("="*70)
print(f"Bounding Box: {KANOMBE_BBOX}")
print(f"Approx coverage: ~4 km × ~4.5 km")
print(f"Expected features: Airport runway, planned residential grid, informal settlements")

# Convert to Earth Engine Geometry
try:
    aoi = ee.Geometry.Rectangle(KANOMBE_BBOX)
    print("[OK] AOI geometry created")
except Exception as e:
    print(f"[X] Failed to create AOI geometry: {e}")
    sys.exit(1)


# ─── Step 3: Load Google Open Buildings V3 ───────────────────────────────────

print("\n[*] Loading Google Open Buildings V3 from Earth Engine...")

try:
    # Dataset: Google Open Buildings V3 (global 50cm resolution building footprints)
    # Paper: https://sites.research.google/open-buildings/
    buildings = ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons")
    print("[OK] Open Buildings V3 dataset loaded")
    
    # Filter to AOI
    buildings_aoi = buildings.filterBounds(aoi)
    print("[OK] Filtered to Kanombe AOI")
    
    # Get count (this triggers a server-side computation)
    print("[...] Counting buildings (this may take 10-20 seconds)...")
    count = buildings_aoi.size().getInfo()
    print(f"[OK] Found {count:,} buildings in Kanombe AOI (before confidence filter)")
    
    if count == 0:
        print("[!] WARNING: Zero buildings found!")
        print("   Possible causes:")
        print("   - Bounding box outside Open Buildings coverage")
        print("   - Dataset access issue")
        print("   Check coverage: https://sites.research.google/open-buildings/")
        sys.exit(1)
    
except Exception as e:
    print(f"[X] Failed to load buildings: {e}")
    print("\nTROUBLESHOOTING:")
    print("   - Verify you have internet connection")
    print("   - Check Earth Engine service status")
    print("   - Try running diagnostic: python gee_diagnostic.py")
    sys.exit(1)


# ─── Step 4: Apply Confidence Filter ─────────────────────────────────────────

# Confidence score: [0.50, 1.00]
# - 0.50-0.70: Low confidence (possible false positives)
# - 0.70-0.85: Medium confidence (recommended threshold)
# - 0.85-1.00: High confidence (verified structures)

CONFIDENCE_THRESHOLD = 0.70

print(f"\n[*] Applying confidence filter: score > {CONFIDENCE_THRESHOLD}")

try:
    buildings_filtered = buildings_aoi.filter(ee.Filter.gte('confidence', CONFIDENCE_THRESHOLD))
    
    print("[...] Counting filtered buildings...")
    count_filtered = buildings_filtered.size().getInfo()
    retention_pct = (count_filtered / count * 100) if count > 0 else 0
    
    print(f"[OK] {count_filtered:,} buildings passed confidence filter ({retention_pct:.1f}% retention)")
    
    if count_filtered == 0:
        print(f"[!] WARNING: All buildings filtered out!")
        print(f"   Try lowering threshold: CONFIDENCE_THRESHOLD = 0.65")
        sys.exit(1)
    
except Exception as e:
    print(f"[X] Confidence filter failed: {e}")
    sys.exit(1)


# ─── Step 5: Export to GeoJSON ───────────────────────────────────────────────

OUTPUT_FILE = "kanombe_buildings.geojson"

print(f"\n[*] Exporting to {OUTPUT_FILE}...")
print("[...] This may take 30-90 seconds depending on building count...")

export_success = False

# Method 1: Try geemap export (faster, cleaner)
try:
    geemap.ee_to_geojson(
        ee_object=buildings_filtered,
        out_json=OUTPUT_FILE,
        timeout=300  # 5 minute timeout
    )
    print(f"[OK] Export complete (geemap method): {OUTPUT_FILE}")
    export_success = True
    
except Exception as e:
    print(f"[!] geemap export failed: {e}")
    print("Trying alternative method...")

# Method 2: Manual export via getInfo() - ONLY works when count <= 5000 (EE limit)
if not export_success and count_filtered <= 5000:
    try:
        print("[...] Downloading building data from Earth Engine...")
        features = buildings_filtered.getInfo()
        
        print("[...] Writing to local file...")
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(features, f)
        
        print(f"[OK] Export complete (manual method): {OUTPUT_FILE}")
        export_success = True
        
    except Exception as e:
        print(f"[!] Manual export failed: {e}")

# Method 3: Chunked local export (bypasses EE's 5000-feature getInfo limit)
if not export_success and count_filtered > 5000:
    print(f"\n[!] Earth Engine blocks direct download > 5000 features.")
    print(f"    You have {count_filtered:,} filtered buildings.")
    print("[*] Performing chunked local export instead (<=5000 features per request)...")

    try:
        # Keep each chunk safely under the 5000 feature limit.
        chunk_size = 2000
        all_features = []
        total = int(count_filtered)

        for start in range(0, total, chunk_size):
            size = min(chunk_size, total - start)
            print(f"   - Fetching chunk: offset={start}, size={size}")

            # Convert the selected slice into a FeatureCollection, then getInfo()
            chunk_list = buildings_filtered.toList(size, start)
            chunk_fc = ee.FeatureCollection(chunk_list)
            chunk_info = chunk_fc.getInfo()

            # chunk_info is a GeoJSON-like dict
            chunk_features = chunk_info.get("features", [])
            all_features.extend(chunk_features)

        geojson_obj = {"type": "FeatureCollection", "features": all_features}
        print(f"[OK] Chunked export assembled: {len(all_features):,} features")

        print("[...] Writing combined GeoJSON to local file...")
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(geojson_obj, f)

        print(f"[OK] Local export complete: {OUTPUT_FILE}")
        export_success = True
    except Exception as e:
        print(f"[X] Chunked export failed: {e}")

if not export_success:
    print(f"[X] FATAL: All export methods failed")
    print("\nTROUBLESHOOTING:")
    print("   - If >5000 buildings: Export to Drive was attempted")
    print("   - Try reducing KANOMBE_BBOX (lines 66-71) for fewer buildings")
    sys.exit(1)

if not export_success:
    print("[X] Export failed")
    sys.exit(1)


# ─── Step 6: Validate Output ──────────────────────────────────────────────────

print("\n" + "="*70)
print("[*] VALIDATION CHECKS")
print("="*70)

try:
    # Load with geopandas to verify
    gdf = gpd.read_file(OUTPUT_FILE)
    
    print(f"\n[OK] Successfully loaded {len(gdf):,} building footprints")
    print(f"   CRS: {gdf.crs}")
    print(f"   Columns: {list(gdf.columns)}")
    
    # Geometry type check
    geom_types = gdf.geometry.type.value_counts()
    print(f"\nGeometry Types:")
    for geom_type, count_geom in geom_types.items():
        pct = count_geom / len(gdf) * 100
        print(f"   {geom_type}: {count_geom:,} ({pct:.1f}%)")
    
    # Confidence distribution
    if 'confidence' in gdf.columns:
        print(f"\nConfidence Score Distribution:")
        print(f"   Mean: {gdf['confidence'].mean():.3f}")
        print(f"   Median: {gdf['confidence'].median():.3f}")
        print(f"   Min: {gdf['confidence'].min():.3f}")
        print(f"   Max: {gdf['confidence'].max():.3f}")
    else:
        print("\n[!] No 'confidence' column found in output")
    
    # Area statistics (quick proxy for feature engineering readiness)
    print("\nComputing area statistics (reprojecting to UTM)...")
    gdf_utm = gdf.to_crs(epsg=32736)  # UTM 36S for Rwanda
    areas = gdf_utm.geometry.area
    
    print(f"\nBuilding Area Statistics (m2):")
    print(f"   Mean: {areas.mean():.1f} m²")
    print(f"   Median: {areas.median():.1f} m²")
    print(f"   Min: {areas.min():.1f} m²")
    print(f"   Max: {areas.max():.1f} m²")
    print(f"   Std Dev: {areas.std():.1f} m²")
    
    # Morphological preview
    small_buildings = len(areas[areas < 50])
    medium_buildings = len(areas[(areas >= 50) & (areas < 200)])
    large_buildings = len(areas[areas >= 200])
    
    print(f"\nMorphological Size Distribution:")
    print(f"   Small (<50 m2): {small_buildings:,} ({small_buildings/len(gdf)*100:.1f}%) - Likely informal")
    print(f"   Medium (50-200 m2): {medium_buildings:,} ({medium_buildings/len(gdf)*100:.1f}%) - Residential")
    print(f"   Large (>200 m2): {large_buildings:,} ({large_buildings/len(gdf)*100:.1f}%) - Commercial/Airport")
    
except Exception as e:
    print(f"[!] Validation failed: {e}")
    print("   File was created but may be corrupted")
    print("   Try re-running the script")


# ─── Step 7: Generate Preview Map ────────────────────────────────────────────

GENERATE_PREVIEW_MAP = True

if GENERATE_PREVIEW_MAP:
    try:
        print("\n[*] Generating preview map...")
        
        import folium
        
        # Calculate center point
        center_lat = gdf.geometry.centroid.y.mean()
        center_lon = gdf.geometry.centroid.x.mean()
        
        print(f"   Map center: ({center_lat:.4f}, {center_lon:.4f})")
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=14,
            tiles='OpenStreetMap'
        )
        
        # Add buildings as GeoJSON layer (sample if >1000 buildings)
        if len(gdf) > 1000:
            print(f"   Sampling 1000 buildings for preview (full dataset has {len(gdf):,})")
            gdf_sample = gdf.sample(n=1000, random_state=42)
        else:
            gdf_sample = gdf
        
        # Add GeoJSON layer
        folium.GeoJson(
            gdf_sample,
            name='Buildings',
            style_function=lambda x: {
                'fillColor': '#3388ff',
                'color': '#3388ff',
                'weight': 1,
                'fillOpacity': 0.4
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['confidence'], 
                aliases=['Confidence:'],
                localize=True
            )
        ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map
        map_file = "kanombe_preview.html"
        m.save(map_file)
        print(f"[OK] Preview map saved: {map_file}")
        print(f"   Open in browser to verify data quality")
        
    except Exception as e:
        print(f"[!] Preview map generation failed: {e}")
        print("   (This is optional - data extraction was successful)")


# ─── Final Summary ────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("[OK] DATA EXTRACTION COMPLETE")
print("="*70)

print(f"\nSUMMARY:")
print(f"   Buildings extracted: {len(gdf):,}")
print(f"   Output file: {OUTPUT_FILE}")
print(f"   Preview map: kanombe_preview.html")

print(f"\nNEXT STEP: Run feature engineering")
print(f"\n   Python code:")
print(f"   ----------------------------------------------------")
print(f"   from feature import compute_features, load_geojson")
print(f"   ")
print(f"   gdf = load_geojson('{OUTPUT_FILE}')")
print(f"   features = compute_features(")
print(f"       gdf,")
print(f"       include_extended=True,")
print(f"       include_orientation=True,")
print(f"       include_multiscale=True")
print(f"   )")
print(f"   ")
print(f"   features.to_csv('kanombe_features_academic.csv', index=False)")
print(f"   print(features[['orientation_entropy', 'density_ratio']].describe())")

print("\n" + "="*70)
print("PROJECT ILLIYA - DATA READY FOR ANALYSIS")
print("="*70)