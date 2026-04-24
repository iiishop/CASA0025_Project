"""
Build the final cached NetworkX graph from the routing-ready network.

Input:
- data/network_routing_input.gpkg

Main output:
- data/main_graph.pkl

This cached graph is what the Flask app loads directly at runtime.
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
ANNA_SCORES_PATH = BASE_DIR / "anna" / "260422_roads_export_final.gpkg"

FOOTPATH_TYPES = {"footway", "path", "pedestrian", "steps"}
FOOTPATH_PENALTY = 1.08
SCORE_COLUMNS = [
    "slope_score",
    "crime_score",
    "air_score",
    "shade_score",
    "wind_score",
    "noise_score",
    "street_activity_score",
    "traffic_score",
]

# Maps raw input columns → normalised output column names
NORM_MAP = {"length_m": "length_norm"}
NORM_MAP.update({col: col.replace("_score", "_norm") for col in SCORE_COLUMNS})


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

for score_col in SCORE_COLUMNS:
    if score_col not in gdf.columns:
        gdf[score_col] = 0.0
    gdf[score_col] = pd.to_numeric(gdf[score_col], errors="coerce").fillna(0.0)

# Inject Anna's composite scores (row-aligned: same base file, same order)
if ANNA_SCORES_PATH.exists():
    print("Loading Anna's composite scores...")
    anna = gpd.read_file(ANNA_SCORES_PATH)[["score_feel_safe", "score_shade_shelter", "score_things_see_do"]]
    assert len(anna) == len(gdf), f"Row count mismatch: Anna {len(anna)} vs routing input {len(gdf)}"
    # crime: high = dangerous → use directly as penalty
    gdf["crime_score"] = anna["score_feel_safe"].values
    # shade: high = good shade → invert so high norm = exposed = penalised
    gdf["shade_score"] = 1.0 - anna["score_shade_shelter"].values
    # street_activity: high = interesting → invert so high norm = boring = penalised
    gdf["street_activity_score"] = 1.0 - anna["score_things_see_do"].values
    print("Anna scores injected: crime_score, shade_score, street_activity_score")
else:
    print(f"Warning: Anna scores not found at {ANNA_SCORES_PATH}, using placeholder zeros.")

scaler = MinMaxScaler()
in_cols = list(NORM_MAP.keys())
out_cols = list(NORM_MAP.values())
gdf[out_cols] = scaler.fit_transform(gdf[in_cols])

if "fclass" not in gdf.columns:
    gdf["fclass"] = pd.NA

gdf["edge_type"] = gdf["fclass"].apply(classify_edge_type)

# Build a graph using the existing OSM u/v node ids
# Each row becomes one edge with length and pre-attached continuous score columns
print(f"Input edges: {len(gdf):,}")
print("Canonical score columns:", SCORE_COLUMNS)
print(gdf[SCORE_COLUMNS].describe())
print("Normalized routing columns:", ["length_norm"] + [col.replace("_score", "_norm") for col in SCORE_COLUMNS])

G = nx.Graph()

for _, row in gdf.iterrows():
    u = int(row["u"])
    v = int(row["v"])
    length_m = float(row["length_m"])
    length_norm = float(row["length_norm"])
    slope_score = float(row["slope_score"])
    crime_score = float(row["crime_score"])
    air_score = float(row["air_score"])
    shade_score = float(row["shade_score"])
    wind_score = float(row["wind_score"])
    noise_score = float(row["noise_score"])
    street_activity_score = float(row["street_activity_score"])
    traffic_score = float(row["traffic_score"])
    slope_norm = float(row["slope_norm"])
    crime_norm = float(row["crime_norm"])
    air_norm = float(row["air_norm"])
    shade_norm = float(row["shade_norm"])
    wind_norm = float(row["wind_norm"])
    noise_norm = float(row["noise_norm"])
    street_activity_norm = float(row["street_activity_norm"])
    traffic_norm = float(row["traffic_norm"])
    edge_type = row["edge_type"]

    type_factor = FOOTPATH_PENALTY if edge_type == "footpath" else 1.0

    edge_data = row.drop(labels="geometry").to_dict()
    edge_data.update({
        "geometry": row.geometry,
        "length_m": length_m,
        "length_norm": length_norm,
        "slope_score": slope_score,
        "crime_score": crime_score,
        "air_score": air_score,
        "shade_score": shade_score,
        "wind_score": wind_score,
        "noise_score": noise_score,
        "street_activity_score": street_activity_score,
        "traffic_score": traffic_score,
        "slope_norm": slope_norm,
        "crime_norm": crime_norm,
        "air_norm": air_norm,
        "shade_norm": shade_norm,
        "wind_norm": wind_norm,
        "noise_norm": noise_norm,
        "street_activity_norm": street_activity_norm,
        "traffic_norm": traffic_norm,
        "edge_type": edge_type,
        "display_type": edge_type,
        "type_factor": type_factor,
        "slope_component": slope_norm,
        "crime_component": crime_norm,
        "air_component": air_norm,
        "shade_component": shade_norm,
        "wind_component": wind_norm,
        "noise_component": noise_norm,
        "street_activity_component": street_activity_norm,
        "traffic_component": traffic_norm,
        "cost_shortest": length_m,
        "cost_easiest": length_norm + slope_norm
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

# Keep only the largest connected component so routing stays stable
largest_cc = components[0]
G_main = G.subgraph(largest_cc).copy()

print(f"Largest connected component: {len(largest_cc):,} nodes")
print(f"Main graph: {G_main.number_of_nodes():,} nodes, {G_main.number_of_edges():,} edges")

with open(GRAPH_CACHE_PATH, "wb") as f:
    pickle.dump(G_main, f)

print(f"Saved graph cache to: {GRAPH_CACHE_PATH}")