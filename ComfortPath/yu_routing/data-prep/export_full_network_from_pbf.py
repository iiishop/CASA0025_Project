"""
Extract the base road network from the Greater London OSM PBF file.
Input:
- greater-london-260401.osm.pbf

Main outputs:
- data/roads_data_full_version.gpkg
- data/osm_walk_nodes.gpkg

This is the first step of the routing data pipeline.
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
from pyrosm import OSM


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


PBF_PATH = BASE_DIR / "greater-london-260401.osm.pbf"
OUT_GPKG = DATA_DIR / "roads_data_full_version.gpkg"
OUT_NODES = DATA_DIR / "osm_walk_nodes.gpkg"
OUT_GEOJSON = DATA_DIR / "roads_data_full_version.geojson"

ROADS_LAYER = "roads_data_full"
NODES_LAYER = "osm_nodes"

EXTRA_ATTRS = [
    "access", "foot", "service", "bridge", "tunnel", "layer",
    "oneway", "junction", "surface", "smoothness", "sidewalk",
    "crossing", "segregated", "lit", "covered", "incline", "ref"
]

BASE_KEEP = {
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "unclassified", "residential", "living_street", "service", "road",
    "pedestrian", "footway", "path", "steps", "track",
    "corridor", "platform", "cycleway", "bridleway"
}

NEVER_WALK = {
    "motorway", "motorway_link",
    "busway", "bus_guideway", "raceway",
    "construction", "proposed"
}

YES_VALUES = {"yes", "designated", "official", "permissive"}
NO_VALUES = {"no", "private"}
SIDEWALK_VALUES = {"yes", "both", "left", "right", "separate"}


def norm_str(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def ensure_col(df: gpd.GeoDataFrame, col: str):
    if col not in df.columns:
        df[col] = pd.NA


print("Loading PBF...")
osm = OSM(str(PBF_PATH))

# Extract both edges and nodes so the original OSM topology is preserved.
print("Extracting raw network with nodes...")
result_a, result_b = osm.get_network(
    network_type="all",
    nodes=True,
    extra_attributes=EXTRA_ATTRS
)

geom_types_a = set(result_a.geometry.geom_type.unique())
if "Point" in geom_types_a or "MultiPoint" in geom_types_a:
    nodes, edges = result_a, result_b
else:
    edges, nodes = result_a, result_b

edges = edges[edges.geometry.notna()].copy()
edges = edges[~edges.geometry.is_empty].copy()
edges = edges.to_crs(27700)

nodes = nodes[nodes.geometry.notna()].copy()
nodes = nodes[~nodes.geometry.is_empty].copy()
nodes = nodes.to_crs(27700)

ensure_col(edges, "name")
for col in [
    "highway", "access", "foot", "service", "bridge", "tunnel", "layer",
    "oneway", "junction", "surface", "smoothness", "sidewalk",
    "crossing", "segregated", "lit", "covered", "incline", "ref"
]:
    ensure_col(edges, col)

for col in [
    "highway", "access", "foot", "service", "bridge", "tunnel", "layer",
    "oneway", "junction", "surface", "smoothness", "sidewalk",
    "crossing", "segregated", "lit", "covered", "incline", "ref"
]:
    edges[col] = norm_str(edges[col])

edges["osm_id"] = edges["id"]
edges["fclass"] = edges["highway"]
edges["length_m"] = edges.geometry.length
edges["source_file"] = PBF_PATH.name

# Keep a broad walkable network at this stage, more specific routing filters are applied later in prepare_routing_input.py.
edges = edges[edges["fclass"].isin(BASE_KEEP | NEVER_WALK)].copy()

edges["foot_yes"] = edges["foot"].isin(YES_VALUES)
edges["foot_no"] = edges["foot"].isin(NO_VALUES)
edges["access_no"] = edges["access"].isin(NO_VALUES)
edges["sidewalk_present"] = edges["sidewalk"].isin(SIDEWALK_VALUES)

edges["walk_candidate"] = True
edges.loc[edges["fclass"].isin(NEVER_WALK), "walk_candidate"] = False
edges.loc[edges["foot_no"], "walk_candidate"] = False
edges.loc[
    edges["access_no"] & ~edges["foot_yes"] & ~edges["sidewalk_present"],
    "walk_candidate"
] = False

edges["topology_level"] = (
    edges["layer"].fillna("0") + "|" +
    edges["bridge"].fillna("no") + "|" +
    edges["tunnel"].fillna("no")
)

edges_out = edges[[
    "osm_id", "u", "v", "fclass", "name", "length_m",
    "access", "foot", "service", "bridge", "tunnel", "layer",
    "oneway", "junction", "surface", "smoothness", "sidewalk",
    "crossing", "segregated", "lit", "covered", "incline", "ref",
    "walk_candidate", "topology_level", "source_file", "geometry"
]].copy()

if OUT_GPKG.exists():
    OUT_GPKG.unlink()
if OUT_NODES.exists():
    OUT_NODES.unlink()

print(f"Saving canonical edges to GPKG layer '{ROADS_LAYER}'...")
edges_out.to_file(OUT_GPKG, layer=ROADS_LAYER, driver="GPKG")

# This GeoJSON is just a lightweight export for sharing.
print("Saving lightweight GeoJSON export...")
geojson_out = edges_out[edges_out["walk_candidate"]].copy().to_crs(4326)
geojson_out[["osm_id", "fclass", "name", "geometry"]].to_file(OUT_GEOJSON, driver="GeoJSON")

print(f"Saving raw nodes for QA/debug to GPKG layer '{NODES_LAYER}'...")
nodes.to_file(OUT_NODES, layer=NODES_LAYER, driver="GPKG")

print("Done.")
print(edges_out["fclass"].value_counts(dropna=False).head(30))
print(edges_out["walk_candidate"].value_counts(dropna=False))
