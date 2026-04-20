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


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


NETWORK_PATH = DATA_DIR / "network_routing_input.gpkg"
NETWORK_LAYER = "network_routing_input"
GRAPH_CACHE_PATH = DATA_DIR / "main_graph.pkl"

FOOTPATH_TYPES = {"footway", "path", "pedestrian", "steps"}
FOOTPATH_PENALTY = 1.08
SLOPE_FACTOR = 4.0


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

gdf["length_m"] = pd.to_numeric(gdf["length_m"], errors="coerce")
gdf = gdf[gdf["length_m"].notna()].copy()

if "slope_pct" not in gdf.columns:
    gdf["slope_pct"] = 0.0
gdf["slope_pct"] = pd.to_numeric(gdf["slope_pct"], errors="coerce").fillna(0.0)

if "fclass" not in gdf.columns:
    gdf["fclass"] = pd.NA

gdf["edge_type"] = gdf["fclass"].apply(classify_edge_type)

# Build a graph using the existing OSM u/v node ids. Each row becomes one edge with length, slope, and routing cost attributes.
print(f"Input edges: {len(gdf):,}")

G = nx.Graph()

for _, row in gdf.iterrows():
    u = int(row["u"])
    v = int(row["v"])
    length_m = float(row["length_m"])
    slope_pct = float(row["slope_pct"])
    edge_type = row["edge_type"]

    slope_penalty = 1.0 + SLOPE_FACTOR * max(slope_pct, 0.0) / 100.0
    type_factor = FOOTPATH_PENALTY if edge_type == "footpath" else 1.0

    edge_data = row.drop(labels="geometry").to_dict()
    edge_data.update({
        "geometry": row.geometry,
        "length_m": length_m,
        "slope_pct": slope_pct,
        "edge_type": edge_type,
        "display_type": edge_type,
        "type_factor": type_factor,
        "cost_shortest": length_m * type_factor,
        "cost_easiest": length_m * slope_penalty * type_factor
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

# Keep only the largest connected component so routing stays stable.
largest_cc = components[0]
G_main = G.subgraph(largest_cc).copy()

print(f"Largest connected component: {len(largest_cc):,} nodes")
print(f"Main graph: {G_main.number_of_nodes():,} nodes, {G_main.number_of_edges():,} edges")

with open(GRAPH_CACHE_PATH, "wb") as f:
    pickle.dump(G_main, f)

print(f"Saved graph cache to: {GRAPH_CACHE_PATH}")