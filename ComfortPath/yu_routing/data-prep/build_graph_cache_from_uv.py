"""
Build the final cached NetworkX graph from the routing-ready network.

Inputs:
- data/network_routing_input.gpkg
- anna/260422_roads_export_with_env_slope.gpkg

Main output:
- data/main_graph.pkl

Derived penalties (higher = worse, used by Dijkstra & overlay)
  safety_penalty          = score_feel_safe
  activity_penalty        = 1 - score_things_see_do
  walking_effort_penalty  = 1 - score_walking_effort
  shade_shelter_penalty   = 1 - score_shade_shelter_final
  air_penalty             = 1 - score_clean_air
  noise_penalty           = 1 - score_not_too_noisy

Routing primitives
  length_m, length_norm
"""

import pickle
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

NETWORK_PATH = DATA_DIR / "network_routing_input.gpkg"
NETWORK_LAYER = "network_routing_input"
GRAPH_CACHE_PATH = DATA_DIR / "main_graph.pkl"
ANNA_SCORES_PATH = BASE_DIR / "anna" / "260422_roads_export_clean_canonical.gpkg"

FOOTPATH_TYPES = {"footway", "path", "pedestrian", "steps"}
FOOTPATH_PENALTY = 1.08

# Only length needs MinMaxScaler normalisation here.
# walking_effort_penalty is derived directly as 1 - score_walking_effort.
ROUTING_NORM_MAP = {
    "length_m": "length_norm",
}


def classify_edge_type(fclass: str) -> str:
    if pd.isna(fclass):
        return "road"
    return "footpath" if str(fclass).lower() in FOOTPATH_TYPES else "road"


print("Loading routing input...")
gdf = gpd.read_file(NETWORK_PATH, layer=NETWORK_LAYER)

if gdf.crs is None:
    raise ValueError("Routing input file has no CRS.")
if gdf.crs.to_epsg() != 27700:
    print("Reprojecting to EPSG:27700...")
    gdf = gdf.to_crs(27700)

required_cols = ["u", "v", "geometry", "length_m"]
missing = [c for c in required_cols if c not in gdf.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

gdf = gdf[gdf.geometry.notna()].copy()
gdf = gdf[~gdf.geometry.is_empty].copy()

gdf["u"] = pd.to_numeric(gdf["u"], errors="coerce")
gdf["v"] = pd.to_numeric(gdf["v"], errors="coerce")
gdf["length_m"] = pd.to_numeric(gdf["length_m"], errors="coerce")
gdf = gdf[gdf["u"].notna() & gdf["v"].notna() & gdf["length_m"].notna()].copy()
gdf["u"] = gdf["u"].astype("int64")
gdf["v"] = gdf["v"].astype("int64")

if "slope_score" not in gdf.columns:
    gdf["slope_score"] = 0.0
gdf["slope_score"] = pd.to_numeric(gdf["slope_score"], errors="coerce").fillna(0.0)

# Compatibility only slope normalisation fallback.
# This preserves the previous routing behaviour if the canonical raw field is absent.
slope_scaler = MinMaxScaler()
gdf["_walking_effort_penalty_from_slope"] = slope_scaler.fit_transform(gdf[["slope_score"]])

#Inject Anna's canonical scores and derive penalties
if ANNA_SCORES_PATH.exists():
    print("Loading Anna's canonical scores...")
    anna = gpd.read_file(ANNA_SCORES_PATH)
    assert len(anna) == len(gdf), (
        f"Row count mismatch: Anna {len(anna)} vs routing input {len(gdf)}"
    )
    missing_anna_cols = [
        col for col in [
            "score_feel_safe",
            "score_things_see_do",
        ]
        if col not in anna.columns
    ]
    if missing_anna_cols:
        raise ValueError(f"Anna scores file is missing required columns: {missing_anna_cols}")

    def _load_canonical_score(
        col_preferred: str,
        fallback_cols: list[str],
        default_value: float,
        fail_if_missing: bool = False,
    ) -> pd.Series:
        source_col = None
        if col_preferred in anna.columns:
            source_col = col_preferred
        else:
            for fb_col in fallback_cols:
                if fb_col in anna.columns:
                    source_col = fb_col
                    print(
                        f"Warning: Anna lacks canonical {col_preferred}. "
                        f"Using fallback column {fb_col} for this build."
                    )
                    break

        if source_col is None:
            if fail_if_missing:
                print(
                    f"Warning: Anna lacks required canonical field {col_preferred}. "
                    "Cannot continue without this field."
                )
                raise ValueError(
                    f"Anna scores file missing required canonical column: {col_preferred}"
                )
            print(
                f"Warning: Anna lacks {col_preferred} (and fallbacks {fallback_cols}). "
                f"Using neutral default {default_value:.2f}."
            )
            return pd.Series(default_value, index=anna.index, dtype=float)

        s = pd.to_numeric(anna[source_col], errors="coerce")
        missing_n = int(s.isna().sum())
        if missing_n > 0:
            print(
                f"Warning: {source_col} has {missing_n:,} null rows; "
                f"filling nulls with neutral default {default_value:.2f}."
            )
            s = s.fillna(default_value)
        return s.clip(0.0, 1.0)

    if "score_walking_effort" in anna.columns:
        anna_walking = pd.to_numeric(anna["score_walking_effort"], errors="coerce")
        missing_walk = int(anna_walking.isna().sum())
        if missing_walk > 0:
            print(
                f"Warning: Anna score_walking_effort has {missing_walk:,} null rows; "
                "falling back to slope-derived compatibility values for those rows."
            )
            anna_walking = anna_walking.fillna(1.0 - gdf["_walking_effort_penalty_from_slope"])
        gdf["score_walking_effort"] = anna_walking.clip(0.0, 1.0).values
        print("Anna score injected: score_walking_effort")
    else:
        print(
            "Warning: Anna scores file lacks score_walking_effort. "
            "Using slope-derived compatibility values to preserve routing behaviour."
        )
        gdf["score_walking_effort"] = (1.0 - gdf["_walking_effort_penalty_from_slope"]).clip(0.0, 1.0)
        print("Walking effort raw score synthesised from slope_score compatibility path")

    # Raw scores stored as-is for overlay reference (higher = better)
    gdf["score_feel_safe"] = pd.to_numeric(anna["score_feel_safe"], errors="coerce").fillna(0.5).clip(0.0, 1.0).values
    gdf["score_things_see_do"] = pd.to_numeric(anna["score_things_see_do"], errors="coerce").fillna(0.5).clip(0.0, 1.0).values
    gdf["score_shade_shelter_final"] = _load_canonical_score(
        "score_shade_shelter_final",
        fallback_cols=[],
        default_value=0.5,
        fail_if_missing=True,
    ).values

    gdf["score_clean_air"] = _load_canonical_score(
        "score_clean_air",
        fallback_cols=[],
        default_value=0.5,
    ).values
    gdf["score_not_too_noisy"] = _load_canonical_score(
        "score_not_too_noisy",
        fallback_cols=[],
        default_value=0.5,
    ).values

    # Penalties for Dijkstra cost and overlay colouring (higher = worse)
    gdf["safety_penalty"] = gdf["score_feel_safe"]
    gdf["activity_penalty"] = 1.0 - gdf["score_things_see_do"]
    gdf["walking_effort_penalty"] = 1.0 - gdf["score_walking_effort"]
    gdf["shade_shelter_penalty"] = 1.0 - gdf["score_shade_shelter_final"]
    gdf["air_penalty"] = 1.0 - gdf["score_clean_air"]
    gdf["noise_penalty"] = 1.0 - gdf["score_not_too_noisy"]

    print(
        "Anna scores injected: score_feel_safe, score_things_see_do, "
        "score_walking_effort, score_shade_shelter_final, "
        "score_clean_air, score_not_too_noisy"
    )
    print(
        "Penalties derived:    safety_penalty, activity_penalty, "
        "walking_effort_penalty, shade_shelter_penalty, air_penalty, noise_penalty"
    )
else:
    print(f"Warning: Anna scores not found at {ANNA_SCORES_PATH}. Using neutral placeholders.")
    for col in (
        "score_feel_safe",
        "score_things_see_do",
        "score_walking_effort",
        "score_shade_shelter_final",
        "score_clean_air",
        "score_not_too_noisy",
    ):
        gdf[col] = 0.5
    for col in (
        "safety_penalty",
        "activity_penalty",
        "walking_effort_penalty",
        "shade_shelter_penalty",
        "air_penalty",
        "noise_penalty",
    ):
        gdf[col] = 0.5

# normalise length and slope
scaler = MinMaxScaler()
in_cols = list(ROUTING_NORM_MAP.keys())
out_cols = list(ROUTING_NORM_MAP.values())
gdf[out_cols] = scaler.fit_transform(gdf[in_cols])
gdf = gdf.drop(columns=["_walking_effort_penalty_from_slope"])

if "fclass" not in gdf.columns:
    gdf["fclass"] = pd.NA
gdf["edge_type"] = gdf["fclass"].apply(classify_edge_type)

print(f"Input edges: {len(gdf):,}")
print(gdf[["slope_score", "score_walking_effort", "walking_effort_penalty", "safety_penalty",
           "activity_penalty", "shade_shelter_penalty", "air_penalty", "noise_penalty", "length_norm"]].describe())

#Build graph
G = nx.Graph()

for _, row in gdf.iterrows():
    u = int(row["u"])
    v = int(row["v"])
    length_m              = float(row["length_m"])
    length_norm           = float(row["length_norm"])
    slope_score           = float(row["slope_score"])
    score_walking_effort  = float(row["score_walking_effort"])
    walking_effort_penalty = float(row["walking_effort_penalty"])
    safety_penalty        = float(row["safety_penalty"])
    activity_penalty      = float(row["activity_penalty"])
    shade_shelter_penalty = float(row["shade_shelter_penalty"])
    air_penalty           = float(row["air_penalty"])
    noise_penalty         = float(row["noise_penalty"])
    score_feel_safe       = float(row["score_feel_safe"])
    score_things_see_do   = float(row["score_things_see_do"])
    score_shade_shelter_final = float(row["score_shade_shelter_final"])
    score_clean_air       = float(row["score_clean_air"])
    score_not_too_noisy   = float(row["score_not_too_noisy"])
    edge_type             = row["edge_type"]

    type_factor = FOOTPATH_PENALTY if edge_type == "footpath" else 1.0

    edge_data = row.drop(labels="geometry").to_dict()
    edge_data.update({
        "geometry": row.geometry,
        "length_m": length_m,
        "length_norm": length_norm,
        "slope_score": slope_score,
        "score_walking_effort": score_walking_effort,
        "walking_effort_penalty": walking_effort_penalty,
        "safety_penalty": safety_penalty,
        "activity_penalty": activity_penalty,
        "shade_shelter_penalty": shade_shelter_penalty,
        "air_penalty": air_penalty,
        "noise_penalty": noise_penalty,
        "score_feel_safe": score_feel_safe,
        "score_things_see_do": score_things_see_do,
        "score_shade_shelter_final": score_shade_shelter_final,
        "score_clean_air": score_clean_air,
        "score_not_too_noisy": score_not_too_noisy,
        "edge_type": edge_type,
        "display_type": edge_type,
        "type_factor": type_factor,
        "cost_shortest": length_m,
        "cost_easiest": length_norm + walking_effort_penalty,
    })

    if G.has_edge(u, v):
        if edge_data["cost_shortest"] < G[u][v]["cost_shortest"]:
            G[u][v].update(edge_data)
    else:
        G.add_edge(u, v, **edge_data)

print(f"Full graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

components = sorted(nx.connected_components(G), key=len, reverse=True)
if not components:
    raise ValueError("Graph is empty.")

largest_cc = components[0]
G_main = G.subgraph(largest_cc).copy()

print(f"Largest connected component: {len(largest_cc):,} nodes")
print(f"Main graph: {G_main.number_of_nodes():,} nodes, {G_main.number_of_edges():,} edges")

with open(GRAPH_CACHE_PATH, "wb") as f:
    pickle.dump(G_main, f)

print(f"Saved graph cache to: {GRAPH_CACHE_PATH}")
