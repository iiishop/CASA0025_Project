import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import geopandas as gpd


def _safe_col(name: str, max_len: int = 50) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return (name or "col")[:max_len]


def transfer_point_values_to_roads_by_matched_name(
    point_gdf: gpd.GeoDataFrame,
    road_gdf: gpd.GeoDataFrame,
    road_name_col: str,
    point_value_cols: Iterable[str],
    target_crs: str = "EPSG:27700",
    max_match_distance: Optional[float] = None,
    prefix: str = "rc_",
    keep_match_columns: bool = True,
) -> gpd.GeoDataFrame:
    """
    Match each point to its nearest road, then aggregate selected point columns
    using median and write values to roads as follows:

    - named roads: aggregate by road name and assign to all segments with that name
    - unnamed roads (None / NaN / '' / 'None'): aggregate only by matched segment,
      not across all unnamed roads

    Returns a copy of road_gdf with added columns.
    """
    if point_gdf.empty:
        raise ValueError("point_gdf is empty")
    if road_gdf.empty:
        raise ValueError("road_gdf is empty")
    if point_gdf.crs is None:
        raise ValueError("point_gdf has no CRS")
    if road_gdf.crs is None:
        raise ValueError("road_gdf has no CRS")
    if road_name_col not in road_gdf.columns:
        raise ValueError(f"'{road_name_col}' not found in road_gdf")

    point_value_cols = list(point_value_cols)
    missing = [c for c in point_value_cols if c not in point_gdf.columns]
    if missing:
        raise ValueError(f"Missing point columns: {missing}")

    points = point_gdf.to_crs(target_crs).copy()
    roads = road_gdf.to_crs(target_crs).copy()

    points = points[points.geometry.notna()].copy()
    roads = roads[roads.geometry.notna()].copy()

    points = points[points.geometry.geom_type.isin(["Point", "MultiPoint"])].copy()
    roads = roads[roads.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()

    if points.empty:
        raise ValueError("No valid point geometries remain")
    if roads.empty:
        raise ValueError("No valid line geometries remain")

    # Keep original road index as stable segment key
    roads = roads.copy()
    roads["_road_seg_id"] = roads.index.astype(str)

    # Clean road names, but do NOT drop unnamed roads
    roads["_road_name_clean"] = roads[road_name_col].astype("string").str.strip()
    roads["_road_name_clean"] = roads["_road_name_clean"].replace(
        ["", "None", "none", "nan", "NaN"],
        pd.NA
    )

    # Needed because sjoin_nearest returns index_right from the right dataframe index
    roads_for_join = roads[[road_name_col, "_road_name_clean", "_road_seg_id", "geometry"]].copy()

    matched = gpd.sjoin_nearest(
        points,
        roads_for_join,
        how="left",
        distance_col="_match_dist"
    ).copy()

    if max_match_distance is not None:
        matched = matched[matched["_match_dist"] <= max_match_distance].copy()

    for col in point_value_cols:
        matched[col] = pd.to_numeric(matched[col], errors="coerce")

    # Build aggregation key:
    # - named roads -> NAME::<road name>
    # - unnamed roads -> SEG::<matched segment id>
    def make_group_key(row):
        road_name = row["_road_name_clean"]
        seg_id = row["_road_seg_id"]

        if pd.notna(road_name):
            return f"NAME::{road_name}"
        if pd.notna(seg_id):
            return f"SEG::{seg_id}"
        return pd.NA

    matched["_group_key"] = matched.apply(make_group_key, axis=1)
    matched = matched[matched["_group_key"].notna()].copy()

    if matched.empty:
        out = roads.copy()
        for col in point_value_cols:
            out[f"{prefix}{_safe_col(col)}_med"] = np.nan
        if not keep_match_columns:
            out = out.drop(columns=["_road_name_clean", "_road_seg_id"], errors="ignore")
        return out

    grouped = (
        matched.groupby("_group_key")[point_value_cols]
        .median(numeric_only=True)
        .reset_index()
    )

    rename_map = {
        col: f"{prefix}{_safe_col(col)}_med"
        for col in point_value_cols
    }
    grouped = grouped.rename(columns=rename_map)

    # Create same key on road layer for joining:
    # - named roads share by name
    # - unnamed roads only join by their own segment id
    def make_road_group_key(row):
        road_name = row["_road_name_clean"]
        seg_id = row["_road_seg_id"]

        if pd.notna(road_name):
            return f"NAME::{road_name}"
        return f"SEG::{seg_id}"

    roads["_group_key"] = roads.apply(make_road_group_key, axis=1)

    out = roads.merge(grouped, on="_group_key", how="left")

    if keep_match_columns:
        counts = (
            matched.groupby("_group_key")
            .size()
            .rename(f"{prefix}n_points")
            .reset_index()
        )
        out = out.merge(counts, on="_group_key", how="left")
        out[f"{prefix}n_points"] = out[f"{prefix}n_points"].fillna(0).astype(int)
    else:
        out = out.drop(columns=["_road_name_clean", "_road_seg_id", "_group_key"], errors="ignore")

    return out