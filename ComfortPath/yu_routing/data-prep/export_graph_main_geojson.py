"""
Export tiled static network overlay data for the web map.

Reads from data/main_graph.pkl and writes:
- routing-web/static/network/tiles/*.geojson
- routing-web/static/network/network_tiles_manifest.json

Each segment carries penalty fields (higher = worse) so the overlay
colour scale is consistent: green = good, red = bad.

Penalty fields exported:
walking_effort_penalty  — slope / walking effort
safety_penalty          — inverse of feel-safe score
activity_penalty        — inverse of things-to-see score
shade_shelter_penalty   — inverse of score_shade_shelter_final
air_penalty             — inverse of score_clean_air
noise_penalty           — inverse of score_not_too_noisy

Raw score fields also exported for tooltip display:
score_feel_safe, score_things_see_do,
score_shade_shelter_final, score_clean_air, score_not_too_noisy
"""

import json
import pickle
from pathlib import Path

import geopandas as gpd
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATIC_NETWORK_DIR = BASE_DIR / "routing-web" / "static" / "network"
TILES_DIR = STATIC_NETWORK_DIR / "tiles"

GRAPH_CACHE_PATH = DATA_DIR / "main_graph.pkl"
MANIFEST_PATH = STATIC_NETWORK_DIR / "network_tiles_manifest.json"

DISPLAY_COLUMNS = [
    "display_id",
    "osm_id",
    # raw scores (higher = better)
    "score_feel_safe",
    "score_things_see_do",
    "score_shade_shelter_final",
    "score_clean_air",
    "score_not_too_noisy",
    # penalties (higher = worse)
    "walking_effort_penalty",
    "safety_penalty",
    "activity_penalty",
    "shade_shelter_penalty",
    "air_penalty",
    "noise_penalty",
    # normalised penalties for overlay colour scale
    "walking_effort_norm",
    "safety_norm",
    "activity_norm",
    "shade_shelter_norm",
    "air_norm",
    "noise_norm",
    "geometry",
]

# penalty_column, label, manifest_key, norm_column
VARIABLE_SPECS = [
    ("walking_effort_penalty", "Walking effort", "walking_effort", "walking_effort_norm"),
    ("safety_penalty", "Safety", "safety", "safety_norm"),
    ("activity_penalty", "Street activity","activity","activity_norm"),
    ("shade_shelter_penalty", "Shade & shelter","shade_shelter", "shade_shelter_norm"),
    ("air_penalty", "Air quality","air", "air_norm"),
    ("noise_penalty","Noise","noise","noise_norm"),
]

SIMPLIFY_TOLERANCE_M = 8.0
TILE_SIZE_DEG = 0.02
DISPLAY_EXCLUDE_FCLASS = {"footway", "path", "steps", "cycleway", "bridleway"}


def minmax_normalize(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").fillna(0.0)
    min_val = float(series.min())
    max_val = float(series.max())
    if max_val > min_val:
        return (series - min_val) / (max_val - min_val)
    return pd.Series(0.0, index=series.index, dtype=float)


def build_display_gdf(graph) -> gpd.GeoDataFrame:
    rows = []

    for _, _, data in graph.edges(data=True):
        fclass = str(data.get("fclass") or "").lower()
        if fclass in DISPLAY_EXCLUDE_FCLASS:
            continue
        rows.append({
            "osm_id":                data.get("osm_id"),
            "length_m":              data.get("length_m"),
            "score_feel_safe":       data.get("score_feel_safe", 0.5),
            "score_things_see_do":   data.get("score_things_see_do", 0.5),
            "score_shade_shelter_final": data.get("score_shade_shelter_final", 0.5),
            "score_clean_air":       data.get("score_clean_air", 0.5),
            "score_not_too_noisy":   data.get("score_not_too_noisy", 0.5),
            "walking_effort_penalty":data.get("walking_effort_penalty", 0.0),
            "safety_penalty":        data.get("safety_penalty", 0.5),
            "activity_penalty":      data.get("activity_penalty", 0.5),
            "shade_shelter_penalty": data.get("shade_shelter_penalty", 0.5),
            "air_penalty":           data.get("air_penalty", 0.0),
            "noise_penalty":         data.get("noise_penalty", 0.0),
            "geometry":              data.get("geometry"),
        })

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:27700")
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    gdf["length_m"] = pd.to_numeric(gdf["length_m"], errors="coerce")
    gdf["display_id"] = gdf["osm_id"].astype("string")
    gdf.loc[gdf["display_id"].isna(), "display_id"] = (
        "segment_" + gdf.loc[gdf["display_id"].isna()].index.astype(str)
    )

    agg_map = {"osm_id": "first", "length_m": "sum"}
    raw_score_cols = [
        "score_feel_safe",
        "score_things_see_do",
        "score_shade_shelter_final",
        "score_clean_air",
        "score_not_too_noisy",
    ]
    for col in raw_score_cols:
        agg_map[col] = "mean"
    for penalty_col, _, _, _ in VARIABLE_SPECS:
        agg_map[penalty_col] = "max"  # worst segment on a shared osm_id wins

    gdf = gdf.dissolve(by="display_id", aggfunc=agg_map, as_index=False)
    gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_TOLERANCE_M, preserve_topology=True)

    for penalty_col, _, _, norm_col in VARIABLE_SPECS:
        gdf[penalty_col] = gdf[penalty_col].fillna(0.0)
        gdf[norm_col] = minmax_normalize(gdf[penalty_col])

    keep_cols = [col for col in DISPLAY_COLUMNS if col in gdf.columns]
    return gdf[keep_cols].copy()


def build_manifest(gdf: gpd.GeoDataFrame, tile_index: dict) -> dict:
    variables = []
    for penalty_col, label, key, norm_col in VARIABLE_SPECS:
        series = gdf[penalty_col]
        variables.append({
            "key": key,
            "label": label,
            "penalty_column": penalty_col,
            "norm_column": norm_col,
            # Overlay weighted mean currently defaults to equal weights.
            # Frontend may override if future UI/API supplies custom weights.
            "default_weight": 1.0,
            "has_variation": bool(float(series.max()) > float(series.min())),
        })

    return {
        "tile_size_deg": TILE_SIZE_DEG,
        "crs": "EPSG:4326",
        "simplify_tolerance_m": SIMPLIFY_TOLERANCE_M,
        "bounds": {
            "min_lon": float(gdf.total_bounds[0]),
            "min_lat": float(gdf.total_bounds[1]),
            "max_lon": float(gdf.total_bounds[2]),
            "max_lat": float(gdf.total_bounds[3]),
        },
        "tiles": tile_index,
        "variables": variables,
        # Defaults for initial overlay selection in UI
        "default_variables": ["walking_effort", "safety", "activity"],
        "overlay_scoring": {
            "value_semantics": "penalty",
            "direction": "higher_is_worse",
            "formula": "sum(w_i * penalty_i_norm) / sum(w_i) over active variables",
            "neutral_state": "if no active variables or denominator==0, score=null",
            "color_mapping": {
                "green": "better",
                "red": "worse",
            },
        },
    }


def add_tile_ids(gdf_4326: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf_4326.copy()
    centroids = out.geometry.representative_point()
    min_lon = float(out.total_bounds[0])
    min_lat = float(out.total_bounds[1])
    out["tile_x"] = ((centroids.x - min_lon) / TILE_SIZE_DEG).apply(int)
    out["tile_y"] = ((centroids.y - min_lat) / TILE_SIZE_DEG).apply(int)
    out["tile_id"] = out.apply(lambda row: f"tile_{row['tile_x']}_{row['tile_y']}", axis=1)
    return out


def export_tiles(gdf_4326: gpd.GeoDataFrame) -> dict:
    tile_index = {}
    for tile_id, tile_gdf in gdf_4326.groupby("tile_id"):
        tile_path = TILES_DIR / f"{tile_id}.geojson"
        tile_export = tile_gdf.drop(columns=["tile_x", "tile_y", "tile_id"]).copy()
        tile_export.to_file(tile_path, driver="GeoJSON")
        minx, miny, maxx, maxy = tile_gdf.total_bounds
        tile_index[tile_id] = {
            "file": tile_path.name,
            "feature_count": int(len(tile_export)),
            "bounds": {
                "min_lon": float(minx),
                "min_lat": float(miny),
                "max_lon": float(maxx),
                "max_lat": float(maxy),
            },
        }
    return tile_index


def main():
    STATIC_NETWORK_DIR.mkdir(parents=True, exist_ok=True)
    TILES_DIR.mkdir(parents=True, exist_ok=True)

    with open(GRAPH_CACHE_PATH, "rb") as f:
        G_main = pickle.load(f)

    display_gdf = build_display_gdf(G_main)
    display_gdf_4326 = display_gdf.to_crs(4326)
    display_gdf_4326 = add_tile_ids(display_gdf_4326)

    tile_index = export_tiles(display_gdf_4326)

    manifest = build_manifest(display_gdf_4326, tile_index)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Exported tiled overlay to:", TILES_DIR)
    print("Manifest written to:", MANIFEST_PATH)
    print("Rows:", len(display_gdf_4326))
    print("Tiles:", len(tile_index))


if __name__ == "__main__":
    main()
