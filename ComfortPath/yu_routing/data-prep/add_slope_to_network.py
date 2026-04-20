"""
Add DEM-based slope information to the base road network.

Input:
- data/roads_data_full_version.gpkg
- data/london_dem.tif

Main output:
- data/network_full_with_slope.gpkg

This is the second step of the routing data pipeline.
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
import numpy as np
import pandas as pd
import rasterio


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


roads_path = DATA_DIR / "roads_data_full_version.gpkg"
roads_layer = "roads_data_full"
dem_path = DATA_DIR / "london_dem.tif"
output_path = DATA_DIR / "network_full_with_slope.gpkg"
output_layer = "network_full_with_slope"

# Load roads
roads = gpd.read_file(roads_path, layer=roads_layer)

# Load DEM
dem = rasterio.open(dem_path)

# Preserve the broad walking network from the extraction step
if "walk_candidate" in roads.columns:
    roads = roads[roads["walk_candidate"].fillna(False)].copy()

roads = roads[roads.geometry.notnull()].copy()
roads = roads[~roads.geometry.is_empty].copy()

# Reproject roads to British National Grid
roads_27700 = roads.to_crs(epsg=27700)
roads_27700["length_m"] = roads_27700.geometry.length

print("Roads ready:", roads_27700.shape)

# Create start and end points
# Slope is estimated from the elevation difference between the start and end of each road segment.
roads_27700["start_pt"] = roads_27700.geometry.apply(lambda geom: geom.interpolate(0))
roads_27700["end_pt"] = roads_27700.geometry.apply(lambda geom: geom.interpolate(geom.length))

start_gdf = gpd.GeoDataFrame(
    roads_27700[["osm_id"]].copy(),
    geometry=roads_27700["start_pt"],
    crs=roads_27700.crs
)

end_gdf = gpd.GeoDataFrame(
    roads_27700[["osm_id"]].copy(),
    geometry=roads_27700["end_pt"],
    crs=roads_27700.crs
)

# Reproject sample points to DEM CRS
start_gdf = start_gdf.to_crs(dem.crs)
end_gdf = end_gdf.to_crs(dem.crs)

start_coords = [(geom.x, geom.y) for geom in start_gdf.geometry]
end_coords = [(geom.x, geom.y) for geom in end_gdf.geometry]

start_vals = list(dem.sample(start_coords))
end_vals = list(dem.sample(end_coords))

start_elev = np.array([val[0] if val is not None else np.nan for val in start_vals], dtype=float)
end_elev = np.array([val[0] if val is not None else np.nan for val in end_vals], dtype=float)

if dem.nodata is not None:
    start_elev[start_elev == dem.nodata] = np.nan
    end_elev[end_elev == dem.nodata] = np.nan

roads_27700["start_elev"] = start_elev
roads_27700["end_elev"] = end_elev
roads_27700["elev_diff"] = roads_27700["end_elev"] - roads_27700["start_elev"]


# Calculate slope and classify
# Preserve all edges for UV/topology integrity, only assign slope values to sufficiently long edges with valid elevation.
elevation_valid = (
    roads_27700["start_elev"].notna() &
    roads_27700["end_elev"].notna()
)

valid_mask = (
    (roads_27700["length_m"] >= 20) &
    elevation_valid
)

roads_27700["slope_pct"] = np.nan

roads_27700.loc[valid_mask, "slope_pct"] = (
    roads_27700.loc[valid_mask, "elev_diff"].abs() /
    roads_27700.loc[valid_mask, "length_m"] * 100
)

# Cap to remove extreme noise
roads_27700.loc[roads_27700["slope_pct"] > 40, "slope_pct"] = np.nan

# Classify slope
# These classes are mainly for interpretation and quick inspection.
def classify_slope(x):
    if pd.isna(x):
        return "unknown"
    elif x < 3:
        return "easy"
    elif x < 5:
        return "moderate"
    else:
        return "steep"


roads_27700["slope_class"] = roads_27700["slope_pct"].apply(classify_slope)

# Drop helper point columns
roads_27700 = roads_27700.drop(columns=["start_pt", "end_pt"])

# Print summary
print(roads_27700[["length_m", "start_elev", "end_elev", "elev_diff", "slope_pct", "slope_class"]].head())
print(roads_27700["slope_pct"].describe())
print(roads_27700["slope_class"].value_counts(dropna=False))

# Export as GPKG
output_file = Path(output_path)
if output_file.exists():
    output_file.unlink()

roads_27700.to_file(output_path, layer=output_layer, driver="GPKG")
print(f"Saved to: {output_path} (layer={output_layer})")


# Lightweight GeoJSON export for sharing.
roads_full = roads_27700[[
    "osm_id",
    "fclass",
    "name",
    "geometry"
]].copy()

print("Fuller roads shape:", roads_full.shape)
print(roads_full["fclass"].value_counts())

roads_full.to_file(
    DATA_DIR / "roads_data_full_version.geojson",
    driver="GeoJSON"
)