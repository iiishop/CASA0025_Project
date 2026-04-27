"""
Merge slope-derived walking-effort fields into Anna's final export gpkg.

Note:
- first try a geometry-derived segment key match in a strict form
- then retry unmatched rows with a more relaxed geometry key
- only if a very small mis-match remains, use aligned row-order as a final fallback
- keep Anna's final export geometry & ids untouched

Runtime contract note:
- score_walking_effort is the canonical raw field (higher = easier / better)
- walking_effort_penalty will later be derived downstream as 1 - score_walking_effort
- slope_score and slope_norm are retained here as engineering / compatibility fields
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.ops import linemerge
from sklearn.preprocessing import MinMaxScaler


BASE_DIR = Path(__file__).resolve().parent
YU_ROUTING_DIR = BASE_DIR.parent
DATA_DIR = YU_ROUTING_DIR / "data"
BASE_DIR = Path(__file__).resolve().parent
YU_ROUTING_DIR = BASE_DIR.parent
DATA_DIR = YU_ROUTING_DIR / "data"
ANNA_DIR = YU_ROUTING_DIR / "anna"

ANNA_PATH = ANNA_DIR / "260422_roads_export_final_with_env.gpkg"
OUR_PATH = DATA_DIR / "network_routing_input.gpkg"
OUT_PATH = ANNA_DIR / "260422_roads_export_with_env_slope.gpkg"
OUT_PATH = ANNA_DIR / "260422_roads_export_with_env_slope.gpkg"
COMMON_CRS = "EPSG:27700"
STRICT_ROUND_DIGITS = 3
RELAXED_ROUND_DIGITS = 1


def _as_line(geom):
    if geom is None or geom.is_empty:
        return None

    if geom.geom_type == "LineString":
        return geom

    if geom.geom_type == "MultiLineString":
        merged = linemerge(geom)
        if merged.geom_type == "LineString":
            return merged
        # keep the longest part if the geometry still cannot merge into one line
        return max(merged.geoms, key=lambda g: g.length)

    return None


def _segment_key(row, round_digits: int, include_length: bool = True) -> str | None:
    geom = _as_line(row.geometry)
    if geom is None:
        return None

    coords = list(geom.coords)
    if len(coords) < 2:
        return None

    start_xy = (round(float(coords[0][0]), round_digits), round(float(coords[0][1]), round_digits))
    end_xy = (round(float(coords[-1][0]), round_digits), round(float(coords[-1][1]), round_digits))

    end_a, end_b = sorted([start_xy, end_xy])
    osm_id = row.get("osm_id")

    key = f"{osm_id}|{end_a[0]}|{end_a[1]}|{end_b[0]}|{end_b[1]}"
    if include_length:
        length_m = round(float(geom.length), round_digits)
        key = f"{key}|{length_m}"

    return key


def _build_unique_lookup(df: gpd.GeoDataFrame, key_col: str) -> pd.DataFrame:
    key_counts = df[key_col].value_counts(dropna=False)
    unique_keys = key_counts[key_counts == 1].index
    out = df[df[key_col].isin(unique_keys)][[key_col, "slope_score"]].copy()
    return out.drop_duplicates(subset=[key_col])


print("Loading Anna's export...")
anna = gpd.read_file(ANNA_PATH)

print("Loading our routing input (slope data)...")
our = gpd.read_file(OUR_PATH, layer="network_routing_input")

print(f"Reprojecting both layers to {COMMON_CRS} for geometry-based comparison...")
anna_match = anna.to_crs(COMMON_CRS).copy()
our_match = our.to_crs(COMMON_CRS).copy()

anna_match["segment_key_strict"] = anna_match.apply(
    lambda row: _segment_key(row, STRICT_ROUND_DIGITS, include_length=True), axis=1
)
our_match["segment_key_strict"] = our_match.apply(
    lambda row: _segment_key(row, STRICT_ROUND_DIGITS, include_length=True), axis=1
)

anna_match["segment_key_relaxed"] = anna_match.apply(
    lambda row: _segment_key(row, RELAXED_ROUND_DIGITS, include_length=False), axis=1
)
our_match["segment_key_relaxed"] = our_match.apply(
    lambda row: _segment_key(row, RELAXED_ROUND_DIGITS, include_length=False), axis=1
)

if anna_match["segment_key_strict"].isna().any():
    raise ValueError(f"Anna export contains {int(anna_match['segment_key_strict'].isna().sum()):,} invalid geometries for key building.")
if our_match["segment_key_strict"].isna().any():
    raise ValueError(f"Routing input contains {int(our_match['segment_key_strict'].isna().sum()):,} invalid geometries for key building.")

anna_dup_strict = int(anna_match["segment_key_strict"].duplicated().sum())
our_dup_strict = int(our_match["segment_key_strict"].duplicated().sum())
anna_dup_relaxed = int(anna_match["segment_key_relaxed"].duplicated().sum())
our_dup_relaxed = int(our_match["segment_key_relaxed"].duplicated().sum())

print(f"Duplicate strict keys  → Anna: {anna_dup_strict:,}, Ours: {our_dup_strict:,}")
print(f"Duplicate relaxed keys → Anna: {anna_dup_relaxed:,}, Ours: {our_dup_relaxed:,}")

strict_lookup = _build_unique_lookup(our_match, "segment_key_strict")
relaxed_lookup = _build_unique_lookup(our_match, "segment_key_relaxed")

our_match["slope_score"] = pd.to_numeric(our_match["slope_score"], errors="coerce").fillna(0.0)
strict_lookup["slope_score"] = pd.to_numeric(strict_lookup["slope_score"], errors="coerce").fillna(0.0)
relaxed_lookup["slope_score"] = pd.to_numeric(relaxed_lookup["slope_score"], errors="coerce").fillna(0.0)

anna_out = anna.copy()
anna_out["segment_key_strict"] = anna_match["segment_key_strict"]
anna_out["segment_key_relaxed"] = anna_match["segment_key_relaxed"]

anna_out = anna_out.merge(
    strict_lookup.rename(columns={"slope_score": "slope_score_strict"}),
    on="segment_key_strict",
    how="left",
)
anna_out = anna_out.merge(
    relaxed_lookup.rename(columns={"slope_score": "slope_score_relaxed"}),
    on="segment_key_relaxed",
    how="left",
)

anna_out["slope_score"] = anna_out["slope_score_strict"]
anna_out.loc[anna_out["slope_score"].isna(), "slope_score"] = anna_out.loc[
    anna_out["slope_score"].isna(), "slope_score_relaxed"
]

fallback_mask = anna_out["slope_score"].isna()
fallback_count = int(fallback_mask.sum())
if fallback_count > 0:
    print(f"Falling back to aligned row-order transfer for {fallback_count:,} residual rows...")
    anna_osm = anna_out.loc[fallback_mask, "osm_id"].reset_index()
    our_osm = our_match.loc[fallback_mask, ["osm_id", "slope_score"]].reset_index(drop=True)
    if (anna_osm["osm_id"].to_numpy() == our_osm["osm_id"].to_numpy()).all():
        anna_out.loc[fallback_mask, "slope_score"] = our_osm["slope_score"].to_numpy()
    else:
        raise ValueError("Residual fallback rows do not preserve aligned osm_id order.")

missing_matches = int(anna_out["slope_score"].isna().sum())
matched_rows = int(len(anna_out) - missing_matches)
print(f"Matched slope rows: {matched_rows:,} / {len(anna_out):,}")

if missing_matches > 0:
    raise ValueError(f"Geometry-based merge left {missing_matches:,} rows without slope_score.")

scaler = MinMaxScaler()
anna_out["slope_norm"] = scaler.fit_transform(anna_out[["slope_score"]])

# Align with Anna's score_ naming scheme; 
# higher = easier (less effort), lower = steeper
anna_out["score_walking_effort"] = 1.0 - anna_out["slope_norm"]

consistency_diff = (anna_out["score_walking_effort"] - (1.0 - anna_out["slope_norm"])).abs()
max_consistency_diff = float(consistency_diff.max())
if max_consistency_diff > 1e-12:
    raise ValueError(
        "score_walking_effort is inconsistent with 1 - slope_norm "
        f"(max abs diff = {max_consistency_diff:.3e})."
    )

anna_out = anna_out.drop(columns=[
    "segment_key_strict",
    "segment_key_relaxed",
    "slope_score_strict",
    "slope_score_relaxed",
])

print(f"slope_score: min={anna_out['slope_score'].min():.2f}  max={anna_out['slope_score'].max():.2f}")
print(f"slope_norm:  min={anna_out['slope_norm'].min():.4f}  max={anna_out['slope_norm'].max():.4f}")
print(f"score_walking_effort: min={anna_out['score_walking_effort'].min():.4f}  max={anna_out['score_walking_effort'].max():.4f}")
print(f"score_walking_effort consistency vs 1 - slope_norm: max_abs_diff={max_consistency_diff:.3e}")
print(f"Segments with slope > 0: {anna_out['slope_score'].gt(0).sum():,} / {len(anna_out):,}")

print(f"\nSaving to {OUT_PATH} ...")
if OUT_PATH.exists():
    OUT_PATH.unlink()
anna_out.to_file(OUT_PATH, driver="GPKG")
print("Done.")
print("\nFinal columns:", list(anna_out.columns))