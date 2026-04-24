"""
Export tiled static network overlay data for the web map.

This export is separate from the routing cache on purpose:
- the graph cache remains focused on fast route solving
- the overlay export stores lightweight segment attributes for map colouring
- the overlay is split into static tiles so the frontend only loads what is visible

Main outputs:
- routing-web/static/network/tiles/*.geojson
- routing-web/static/network/network_tiles_manifest.json
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
    "slope_score",
    "crime_score",
    "air_score",
    "shade_score",
    "wind_score",
    "noise_score",
    "street_activity_score",
    "traffic_score",
    "slope_norm",
    "crime_score_norm",
    "air_score_norm",
    "shade_score_norm",
    "wind_score_norm",
    "noise_score_norm",
    "street_activity_score_norm",
    "traffic_score_norm",
    "geometry",
]

# (raw_column, label, manifest_key, norm_column)
VARIABLE_SPECS = [
    ("slope_score", "Slope", "slope", "slope_norm"),
    ("crime_score", "Crime", "crime", "crime_score_norm"),
    ("air_score", "Air", "air", "air_score_norm"),
    ("shade_score", "Shade", "shade", "shade_score_norm"),
    ("wind_score", "Wind", "wind", "wind_score_norm"),
    ("noise_score", "Noise", "noise", "noise_score_norm"),
    ("street_activity_score", "Street activity", "street_activity", "street_activity_score_norm"),
    ("traffic_score", "Traffic", "traffic", "traffic_score_norm"),
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

    for u, v, data in graph.edges(data=True):
        fclass = str(data.get("fclass") or "").lower()
        if fclass in DISPLAY_EXCLUDE_FCLASS:
            continue
        rows.append({
            "osm_id": data.get("osm_id"),
            "length_m": data.get("length_m"),
            "slope_score": data.get("slope_score", 0.0),
            "crime_score": data.get("crime_score", 0.0),
            "air_score": data.get("air_score", 0.0),
            "shade_score": data.get("shade_score", 0.0),
            "wind_score": data.get("wind_score", 0.0),
            "noise_score": data.get("noise_score", 0.0),
            "street_activity_score": data.get("street_activity_score", 0.0),
            "traffic_score": data.get("traffic_score", 0.0),
            "geometry": data.get("geometry"),
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
    for score_col, _, _, _ in VARIABLE_SPECS:
        agg_map[score_col] = "max"

    gdf = gdf.dissolve(by="display_id", aggfunc=agg_map, as_index=False)
    gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_TOLERANCE_M, preserve_topology=True)

    for score_col, _, _, norm_col in VARIABLE_SPECS:
        gdf[score_col] = gdf[score_col].fillna(0.0)
        gdf[norm_col] = minmax_normalize(gdf[score_col])

    keep_cols = [col for col in DISPLAY_COLUMNS if col in gdf.columns]
    return gdf[keep_cols].copy()


def build_manifest(gdf: gpd.GeoDataFrame, tile_index: dict[str, dict]) -> dict:
    variables = []

    for score_col, label, key, norm_col in VARIABLE_SPECS:
        score_series = gdf[score_col]
        variables.append({
            "key": key,
            "label": label,
            "raw_column": score_col,
            "norm_column": norm_col,
            "has_variation": bool(float(score_series.max()) > float(score_series.min())),
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
        "default_variables": ["slope"],
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


def export_tiles(gdf_4326: gpd.GeoDataFrame) -> dict[str, dict]:
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

    print("Exported tiled static network overlay data to:", TILES_DIR)
    print("Exported overlay manifest to:", MANIFEST_PATH)
    print("Rows:", len(display_gdf_4326))
    print("Tiles:", len(tile_index))


if __name__ == "__main__":
    main()