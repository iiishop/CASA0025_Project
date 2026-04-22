"""
Prepare the final routing network from the added slope road dataset.

Input:
- data/network_full_with_slope.gpkg

Main output:
- data/network_routing_input.gpkg

This step applies walkability and routing-specific filtering rules.
"""

import os
import sys
from pathlib import Path


conda_prefix = Path(sys.prefix)
proj_dir = conda_prefix / "Library" / "share" / "proj"
gdal_dir = conda_prefix / "Library" / "share" / "gdal"

if not (proj_dir / "proj.db").exists():
    raise FileNotFoundError(f"PROJ database not found: {proj_dir / 'proj.db'}")
if not gdal_dir.exists():
    raise FileNotFoundError(f"GDAL data dir not found: {gdal_dir}")

os.environ.pop("PROJ_LIB", None)
os.environ.pop("GDAL_DATA", None)
os.environ["PROJ_LIB"] = str(proj_dir)
os.environ["GDAL_DATA"] = str(gdal_dir)

from pyproj.datadir import set_data_dir

set_data_dir(str(proj_dir))

import geopandas as gpd
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


INPUT_PATH = DATA_DIR / "network_full_with_slope.gpkg"
INPUT_LAYER = "network_full_with_slope"
OUTPUT_PATH = DATA_DIR / "network_routing_input.gpkg"
OUTPUT_LAYER = "network_routing_input"

YES_VALUES = {"yes", "designated", "official", "permissive"}
SIDEWALK_VALUES = {"yes", "both", "left", "right", "separate"}
EXCLUDED_SERVICE_VALUES = {"parking_aisle", "driveway", "drive-through", "emergency_access"}

BACKBONE_TYPES = {
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "unclassified", "residential", "living_street", "road",
    "pedestrian"
}


def ensure_col(df: gpd.GeoDataFrame, col: str):
    if col not in df.columns:
        df[col] = pd.NA


def norm_str(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


print("Loading slope-enriched full network...")
gdf = gpd.read_file(INPUT_PATH, layer=INPUT_LAYER)

gdf = gdf[gdf.geometry.notna()].copy()
gdf = gdf[~gdf.geometry.is_empty].copy()

if gdf.crs is None:
    raise ValueError("Input routing network file has no CRS.")
if gdf.crs.to_epsg() != 27700:
    print("Reprojecting to EPSG:27700...")
    gdf = gdf.to_crs(27700)

for col in ["fclass", "foot", "service", "sidewalk", "crossing", "segregated"]:
    ensure_col(gdf, col)
    gdf[col] = norm_str(gdf[col])

if "length_m" not in gdf.columns:
    gdf["length_m"] = gdf.geometry.length
gdf["length_m"] = pd.to_numeric(gdf["length_m"], errors="coerce")
gdf = gdf[gdf["length_m"].notna()].copy()

for score_col in ["slope_score", "crime_score", "air_score"]:
    ensure_col(gdf, score_col)
    gdf[score_col] = pd.to_numeric(gdf[score_col], errors="coerce").fillna(0.0)

for node_col in ["u", "v"]:
    if node_col not in gdf.columns:
        raise ValueError(f"Input network is missing required routing column: {node_col}")
    gdf[node_col] = pd.to_numeric(gdf[node_col], errors="coerce")

gdf = gdf[gdf["u"].notna() & gdf["v"].notna()].copy()
gdf["u"] = gdf["u"].astype("int64")
gdf["v"] = gdf["v"].astype("int64")

ensure_col(gdf, "osm_id")
ensure_col(gdf, "fclass")

if "walk_candidate" in gdf.columns:
    gdf = gdf[gdf["walk_candidate"].fillna(False)].copy()

# Define the main routing inclusion rules
# The aim is to keep a realistic walking network without making it too noisy
fclass = gdf["fclass"]
length_m = gdf["length_m"]
foot_yes = gdf["foot"].isin(YES_VALUES)
sidewalk_present = gdf["sidewalk"].isin(SIDEWALK_VALUES)
crossing_present = gdf["crossing"].notna() & (gdf["crossing"] != "no")
shared_cycleway = (gdf["fclass"] == "cycleway") & (foot_yes | (gdf["segregated"] == "no"))

mask_backbone = fclass.isin(BACKBONE_TYPES)

mask_footway = (fclass == "footway")

mask_service = (
    (fclass == "service")
    & ~gdf["service"].isin(EXCLUDED_SERVICE_VALUES)
)

mask_path = (
    (fclass == "path")
    & (foot_yes | crossing_present | (length_m >= 5))
)

mask_cycleway = shared_cycleway

gdf["routing_reason"] = pd.NA
gdf.loc[mask_backbone, "routing_reason"] = "backbone"
gdf.loc[mask_footway & gdf["routing_reason"].isna(), "routing_reason"] = "footway_or_crossing"
gdf.loc[mask_service & gdf["routing_reason"].isna(), "routing_reason"] = "service_filtered"
gdf.loc[mask_path & gdf["routing_reason"].isna(), "routing_reason"] = "path_filtered"
gdf.loc[mask_cycleway & gdf["routing_reason"].isna(), "routing_reason"] = "shared_cycleway"

gdf["include_in_routing"] = (
    mask_backbone |
    mask_footway |
    mask_service |
    mask_path |
    mask_cycleway
)

routing_gdf = gdf[gdf["include_in_routing"]].copy()
routing_gdf = routing_gdf[routing_gdf.geometry.length > 0].copy()
routing_gdf["length_m"] = routing_gdf.geometry.length

# Export the final network that will actually be used to build the graph
print(f"Full network edges: {len(gdf):,}")
print(f"Routing input edges: {len(routing_gdf):,}")
print("Included fclass counts:")
print(routing_gdf["fclass"].value_counts(dropna=False).head(30))
print("Routing reason counts:")
print(gdf.loc[gdf["include_in_routing"], "routing_reason"].value_counts(dropna=False))
print("Canonical score columns:", ["slope_score", "crime_score", "air_score"])
print(routing_gdf[["slope_score", "crime_score", "air_score"]].describe())

# Canonical road-segment score schema first, followed by routing-specific fields
final_cols = [
    "osm_id",
    "geometry",
    "length_m",
    "slope_score",
    "crime_score",
    "air_score",
    "u",
    "v",
    "fclass",
]
final_cols = [col for col in final_cols if col in routing_gdf.columns]
routing_gdf = routing_gdf[final_cols].copy()

print("Final routing export columns:", final_cols)

if OUTPUT_PATH.exists():
    OUTPUT_PATH.unlink()

print(f"Saving routing input to {OUTPUT_PATH} (layer={OUTPUT_LAYER})...")
routing_gdf.to_file(OUTPUT_PATH, layer=OUTPUT_LAYER, driver="GPKG")
print("Done.")