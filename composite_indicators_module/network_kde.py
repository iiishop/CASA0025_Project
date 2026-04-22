import math
import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import geopandas as gpd
"""
This script computes line-based kernel density estimates (KDE) by projecting point events onto nearby line segments. It is designed for street-network style analysis where points represent observations or amenities, and lines represent street segments or other linear features.

The main function, network_kde_on_lines, returns a copy of the input line layer with new columns describing:

- KDE intensity per segment
- raw nearby point counts
- optional weighted counts
- optional normalized ranks from 0 to 1

The output keeps all original line segments. Segments that are too short for analysis are retained and filled with zero-valued outputs.
"""

def _safe_field_name(name: str, max_len: int = 40) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return (name or "unknown")[:max_len]


def _kde_weight(dist: float, bandwidth: float, kernel: str = "gaussian") -> float:
    if dist > bandwidth:
        return 0.0
    if kernel == "gaussian":
        return math.exp(-0.5 * (dist / bandwidth) ** 2)
    if kernel == "quartic":
        return (1 - (dist / bandwidth) ** 2) ** 2
    raise ValueError("kernel must be 'gaussian' or 'quartic'")


def _rank_0_1(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    min_val = s.min()
    max_val = s.max()

    if pd.isna(min_val) or pd.isna(max_val):
        return pd.Series(np.nan, index=series.index)

    if max_val == min_val:
        return pd.Series(0.0, index=series.index)

    return (s - min_val) / (max_val - min_val)


def network_kde_on_lines(
    point_gdf: gpd.GeoDataFrame,
    line_gdf: gpd.GeoDataFrame,
    value_name: str = "points",
    category_col: Optional[str] = None,
    mode: str = "all_only",
    bandwidth: float = 100.0,
    kernel: str = "gaussian",
    target_crs: str = "EPSG:27700",
    min_segment_length: float = 0.0,
    clip_polygon_gdf: Optional[gpd.GeoDataFrame] = None,
    categories: Optional[Sequence[str]] = None,
    include_counts: bool = True,
    include_weighted_counts: bool = False,
    include_ranks: bool = True,
    weight_col: Optional[str] = None,
    allow_negative_weights: bool = False,
    length_col: str = "segment_length",
    eligible_col: str = "kde_eligible",
) -> gpd.GeoDataFrame:
    """
    Compute line-based KDE from points onto line segments.

    Returns a copy of the full input line_gdf in target_crs, with KDE/count/rank
    columns added. Segments below min_segment_length are kept and filled with 0.

    Parameters
    ----------
    weight_col : str, optional
        Numeric column in point_gdf used to weight each point's contribution.
        If None, each point has weight 1.0.
    include_weighted_counts : bool
        If True, add weighted_count_* fields storing the sum of point weights
        within bandwidth for each segment.
    allow_negative_weights : bool
        If False, negative weights raise an error.
    """
    if point_gdf.empty:
        raise ValueError("point_gdf is empty")
    if line_gdf.empty:
        raise ValueError("line_gdf is empty")
    if point_gdf.crs is None:
        raise ValueError("point_gdf has no CRS")
    if line_gdf.crs is None:
        raise ValueError("line_gdf has no CRS")
    if mode not in {"all_only", "all_and_categories"}:
        raise ValueError("mode must be 'all_only' or 'all_and_categories'")
    if kernel not in {"gaussian", "quartic"}:
        raise ValueError("kernel must be 'gaussian' or 'quartic'")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be > 0")

    base_name = _safe_field_name(value_name)

    if point_gdf.crs != line_gdf.crs:
        raise ValueError(f"CRS mismatch: point_gdf={point_gdf.crs}, line_gdf={line_gdf.crs}")

    if target_crs is None:
        points = point_gdf.copy()
        lines = line_gdf.copy()
    else:
        points = point_gdf.copy() if point_gdf.crs == target_crs else point_gdf.to_crs(target_crs).copy()
        lines = line_gdf.copy() if line_gdf.crs == target_crs else line_gdf.to_crs(target_crs).copy()

    points = points[points.geometry.notna()].copy()
    lines = lines[lines.geometry.notna()].copy()

    points = points[points.geometry.geom_type.isin(["Point", "MultiPoint"])].copy()
    lines = lines[lines.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()

    if points.empty:
        raise ValueError("point_gdf has no valid point geometries after filtering")
    if lines.empty:
        raise ValueError("line_gdf has no valid line geometries after filtering")

    if clip_polygon_gdf is not None:
        if clip_polygon_gdf.crs is None:
            raise ValueError("clip_polygon_gdf has no CRS")
        clip_poly = clip_polygon_gdf.to_crs(target_crs)
        lines = gpd.clip(lines, clip_poly)
        points = gpd.clip(points, clip_poly)

        lines = lines[lines.geometry.notna()].copy()
        points = points[points.geometry.notna()].copy()

        if lines.empty:
            raise ValueError("No line features remain after clipping")

    if weight_col is not None:
        if weight_col not in points.columns:
            raise ValueError(f"'{weight_col}' not found in point_gdf")

        points = points.copy()
        points["_kde_point_weight"] = pd.to_numeric(points[weight_col], errors="coerce").fillna(0.0)

        if not allow_negative_weights and (points["_kde_point_weight"] < 0).any():
            raise ValueError(f"Negative values found in '{weight_col}' but allow_negative_weights=False")
    else:
        points = points.copy()
        points["_kde_point_weight"] = 1.0

    out = lines.copy()
    out[length_col] = out.geometry.length
    out[eligible_col] = out[length_col] >= min_segment_length

    analysis_groups = {base_name: None}

    if mode == "all_and_categories":
        if not category_col:
            raise ValueError("category_col is required when mode='all_and_categories'")
        if category_col not in points.columns:
            raise ValueError(f"'{category_col}' not found in point_gdf")

        if categories is None:
            raw_categories = sorted(
                c for c in points[category_col].dropna().astype(str).unique()
            )
        else:
            raw_categories = [str(c) for c in categories]

        for cat in raw_categories:
            suffix = f"{base_name}_{_safe_field_name(cat)}"
            analysis_groups[suffix] = cat

    for suffix in analysis_groups:
        if include_counts:
            out[f"count_{suffix}"] = 0
        if include_weighted_counts:
            out[f"weighted_count_{suffix}"] = 0.0
        out[f"kde_{suffix}"] = 0.0
        if include_ranks:
            out[f"rank_{suffix}"] = 0.0

    if points.empty:
        return out

    sindex = points.sindex
    eligible_idx = out.index[out[eligible_col]].tolist()

    for i, idx in enumerate(eligible_idx, start=1):
        line_row = out.loc[idx]
        geom = line_row.geometry
        seg_len = line_row[length_col]

        if seg_len <= 0:
            continue

        minx, miny, maxx, maxy = geom.bounds
        bbox = (minx - bandwidth, miny - bandwidth, maxx + bandwidth, maxy + bandwidth)
        candidate_idx = list(sindex.intersection(bbox))

        if not candidate_idx:
            continue

        candidates = points.iloc[candidate_idx]

        kde_sums = {suffix: 0.0 for suffix in analysis_groups}
        counts = {suffix: 0 for suffix in analysis_groups}
        weighted_counts = {suffix: 0.0 for suffix in analysis_groups}

        for _, point_row in candidates.iterrows():
            dist = geom.distance(point_row.geometry)
            if dist > bandwidth:
                continue

            point_weight = float(point_row["_kde_point_weight"])
            kernel_weight = _kde_weight(dist, bandwidth, kernel)
            contribution = kernel_weight * point_weight

            kde_sums[base_name] += contribution
            counts[base_name] += 1
            weighted_counts[base_name] += point_weight

            if mode == "all_and_categories":
                cat_val = point_row.get(category_col)
                if pd.notna(cat_val):
                    suffix = f"{base_name}_{_safe_field_name(str(cat_val))}"
                    if suffix in kde_sums:
                        kde_sums[suffix] += contribution
                        counts[suffix] += 1
                        weighted_counts[suffix] += point_weight

        for suffix in analysis_groups:
            if include_counts:
                out.at[idx, f"count_{suffix}"] = counts[suffix]
            if include_weighted_counts:
                out.at[idx, f"weighted_count_{suffix}"] = weighted_counts[suffix]
            out.at[idx, f"kde_{suffix}"] = kde_sums[suffix] / seg_len

        if i % 1000 == 0:
            print(f"Processed {i} eligible line segments...")

    if include_ranks:
        for suffix in analysis_groups:
            out[f"rank_{suffix}"] = _rank_0_1(out[f"kde_{suffix}"])

    return out