from pathlib import Path
import geopandas as gpd

# paths
INPUT_PATH = Path(r"D:\casa0025_slope\anna\260422_roads_export_with_env_slope.gpkg")
OUTPUT_PATH = Path(r"D:\casa0025_slope\anna\260422_roads_export_clean_canonical.gpkg")

#columns to keep
KEEP_COLS = [
    "osm_id",
    "score_feel_safe",
    "score_things_see_do",
    "score_walking_effort",
    "score_shade_shelter", # keep because current graph builder still uses it
    "score_shade_shelter_final", # keep because this is likely the true merged final field
    "score_clean_air",
    "score_not_too_noisy",
    "slope_score", # temporary compatibility field
    "geometry",
]

print("Loading input gpkg...")
gdf = gpd.read_file(INPUT_PATH)

print(f"Rows: {len(gdf):,}")
print(f"CRS: {gdf.crs}")

missing = [c for c in KEEP_COLS if c not in gdf.columns]
if missing:
    raise ValueError(f"Missing required columns in input file: {missing}")

clean = gdf[KEEP_COLS].copy()

# enforce a clean column order
clean = clean[
    [
        "osm_id",
        "score_feel_safe",
        "score_things_see_do",
        "score_walking_effort",
        "score_shade_shelter",
        "score_shade_shelter_final",
        "score_clean_air",
        "score_not_too_noisy",
        "slope_score",
        "geometry",
    ]
]

print("\n=== CLEAN FILE SUMMARY ===")
print("columns:", list(clean.columns))
print("rows:", len(clean))
print("null geometry:", int(clean.geometry.isna().sum()))
print("empty geometry:", int(clean.geometry.is_empty.sum()))

if "our_uid" in clean.columns:
    print("our_uid unique:", clean["our_uid"].nunique(dropna=True), "/", len(clean))
if "osm_id" in clean.columns:
    print("osm_id unique:", clean["osm_id"].nunique(dropna=True), "/", len(clean))

# overwrite safely
if OUTPUT_PATH.exists():
    OUTPUT_PATH.unlink()

print(f"\nSaving clean canonical gpkg to:\n{OUTPUT_PATH}")
clean.to_file(OUTPUT_PATH, driver="GPKG")

print("Done.")