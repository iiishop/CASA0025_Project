"""
Core graph and routing functions for the Flask prototype.

This file contains:
- network cleaning helpers
- graph construction
- base edge cost definition
- personalised weight function
- shortest path solving
"""

import warnings
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import pandas as pd
import networkx as nx
import numpy as np

from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union, linemerge
from scipy.spatial import cKDTree


warnings.filterwarnings("ignore", category=UserWarning)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

#functions

def ensure_linestrings(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Keep only linear features and explode MultiLineString geometries into
    individual LineString features.
    """
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf = gdf[gdf.geometry.geom_type == "LineString"].copy()
    return gdf.reset_index(drop=True)


def round_coord(x, y, ndigits=3):
    """
    Standardize node keys to avoid floating point precision issues.
    In EPSG:27700, millimetre precision is unnecessary, so 3 decimals are enough.
    """
    return (round(float(x), ndigits), round(float(y), ndigits))


def line_endpoints(line: LineString):
    coords = list(line.coords)
    return Point(coords[0]), Point(coords[-1])


def replace_line_endpoints(line: LineString, start_xy, end_xy) -> LineString:
    """
    Replace only the first and last vertices of a line while keeping the
    intermediate shape unchanged.
    """
    coords = list(line.coords)
    if len(coords) < 2:
        return line
    new_coords = [start_xy] + coords[1:-1] + [end_xy]
    # Avoid duplicated vertices or degenerate lines after endpoint snapping
    cleaned = [new_coords[0]]
    for c in new_coords[1:]:
        if c != cleaned[-1]:
            cleaned.append(c)
    if len(cleaned) < 2:
        return line
    return LineString(cleaned)


#merge nearby endpoints to one representative point

def snap_nearby_endpoints(gdf: gpd.GeoDataFrame, tolerance=3.0) -> gpd.GeoDataFrame:
    """
    Cluster endpoints within the given tolerance (in metres) and replace them
    with the cluster centroid.
    Only endpoints are snapped, the full line geometry is not shifted.
    """
    gdf = gdf.copy().reset_index(drop=True)

    endpoints = []
    endpoint_meta = []  # (row_id, which_end)
    for idx, geom in enumerate(gdf.geometry):
        p0, p1 = line_endpoints(geom)
        endpoints.append((p0.x, p0.y))
        endpoint_meta.append((idx, "start"))
        endpoints.append((p1.x, p1.y))
        endpoint_meta.append((idx, "end"))

    arr = np.array(endpoints)
    tree = cKDTree(arr)

    # Use BFS & union-like grouping to find connected components
    visited = np.zeros(len(arr), dtype=bool)
    groups = []

    for i in range(len(arr)):
        if visited[i]:
            continue
        queue = [i]
        visited[i] = True
        group = [i]

        while queue:
            cur = queue.pop()
            neighbors = tree.query_ball_point(arr[cur], r=tolerance)
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
                    group.append(nb)

        groups.append(group)

    # Use the centroid of each group as the snapped coordinate
    snapped_xy = {}
    for group in groups:
        xs = arr[group, 0]
        ys = arr[group, 1]
        cx = float(xs.mean())
        cy = float(ys.mean())
        for idx in group:
            snapped_xy[idx] = (cx, cy)

    # Replace endpoints for every line
    new_geoms = []
    for row_id, geom in enumerate(gdf.geometry):
        start_idx = 2 * row_id
        end_idx = 2 * row_id + 1
        start_xy = snapped_xy[start_idx]
        end_xy = snapped_xy[end_idx]
        new_geom = replace_line_endpoints(geom, start_xy, end_xy)
        new_geoms.append(new_geom)

    gdf["geometry"] = new_geoms
    return gdf

# Split the network at intersections

def extract_noded_segments(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Apply unary_union to the whole line network so intersections are split automatically, then extract the resulting individual segments.
    Attributes are lost in this step and must be mapped back later.
    """
    merged = unary_union(gdf.geometry.tolist())

    segments = []

    def collect_lines(geom):
        if geom.is_empty:
            return
        if geom.geom_type == "LineString":
            if geom.length > 0:
                segments.append(geom)
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                if part.length > 0:
                    segments.append(part)
        elif hasattr(geom, "geoms"):
            for part in geom.geoms:
                collect_lines(part)

    collect_lines(merged)

    seg_gdf = gpd.GeoDataFrame(
        {"geometry": segments},
        geometry="geometry",
        crs=gdf.crs
    ).reset_index(drop=True)

    seg_gdf["seg_id"] = seg_gdf.index
    seg_gdf["length_m"] = seg_gdf.geometry.length
    return seg_gdf

# Inherit source attributes for split segments
def map_attributes_to_segments(
    original_gdf: gpd.GeoDataFrame,
    seg_gdf: gpd.GeoDataFrame,
    attr_cols=None
) -> gpd.GeoDataFrame:
    """
    For each split segment, find the original line with the greatest overlap length and inherit its attributes.
    """
    if attr_cols is None:
        attr_cols = [c for c in original_gdf.columns if c != "geometry"]

    original_gdf = original_gdf.copy().reset_index(drop=True)
    seg_gdf = seg_gdf.copy().reset_index(drop=True)

    sindex = original_gdf.sindex

    mapped_rows = []

    for _, seg in seg_gdf.iterrows():
        geom = seg.geometry
        candidate_idx = list(sindex.intersection(geom.bounds))

        best_idx = None
        best_overlap = -1.0

        for idx in candidate_idx:
            orig_geom = original_gdf.iloc[idx].geometry
            try:
                overlap = geom.intersection(orig_geom).length
            except Exception:
                overlap = 0.0

            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx

        row_data = seg.to_dict()

        if best_idx is not None:
            for col in attr_cols:
                row_data[col] = original_gdf.iloc[best_idx][col]

        mapped_rows.append(row_data)

    out = gpd.GeoDataFrame(mapped_rows, geometry="geometry", crs=seg_gdf.crs)

    # Replace length_m with the actual split segment length for robustness
    out["length_m"] = out.geometry.length

    # Default missing slope_pct values to 0
    if "slope_pct" not in out.columns:
        out["slope_pct"] = 0.0
    out["slope_pct"] = pd.to_numeric(out["slope_pct"], errors="coerce").fillna(0.0)

    return out


# Graph construction
def build_graph_from_segments(
    seg_gdf: gpd.GeoDataFrame,
    footpath_penalty: float = 1.08,
    slope_factor: float = 1.0
) -> nx.Graph:
    """
    Each segment becomes one undirected edge.
    Nodes are defined by the start and end coordinates of each segment.

    cost_shortest:
        length * type penalty only
    cost_easiest:
        length * slope penalty * type penalty

    Design logic:
    - road is the default network, factor = 1.0
    - footpath gets a slight penalty so it is not chosen by default when it is nearly equivalent to a road.
    - if a footpath offers a meaningful shortcut (for example through a park), it can still be selected
    """
    G = nx.Graph()

    for _, row in seg_gdf.iterrows():
        geom = row.geometry
        coords = list(geom.coords)
        u = round_coord(*coords[0])
        v = round_coord(*coords[-1])

        length_m = float(row["length_m"])
        slope_pct = float(row.get("slope_pct", 0.0))
        edge_type = row.get("edge_type", "road")

        # Penalise uphill only, use slope_pct later if both directions should be penalised
        slope_penalty = 1.0 + slope_factor * max(slope_pct, 0.0) / 100.0

        # Prefer roads, with a slight penalty on footpaths
        type_factor = footpath_penalty if edge_type == "footpath" else 1.0

        edge_data = row.drop(labels="geometry").to_dict()
        edge_data.update({
            "geometry": geom,
            "length_m": length_m,
            "slope_pct": slope_pct,
            "edge_type": edge_type,
            "display_type": edge_type,
            "type_factor": type_factor,
            "cost_shortest": length_m * type_factor,
            "cost_easiest": length_m * slope_penalty * type_factor
        })

        if G.has_edge(u, v):
            # For duplicate edges, keep the one with the lower shortest cost
            if edge_data["cost_shortest"] < G[u][v]["cost_shortest"]:
                G[u][v].update(edge_data)
        else:
            G.add_edge(u, v, **edge_data)

    return G



# Nearest-node snapping
def build_node_kdtree(G: nx.Graph):
    """
    Build a KDTree for nearest-node lookup on the graph.
    The older version assumed that G.nodes themselves were (x, y) coordinates.
    In the newer graph, nodes are OSM node ids (integers), so node coordinates must be derived from edge geometries.
    """
    node_coords = {}

    for u, v, data in G.edges(data=True):
        geom = data.get("geometry")

        if geom is None:
            continue

        coords = list(geom.coords)
        if len(coords) < 2:
            continue

        # Use the first coordinate of the edge for node u
        if u not in node_coords:
            node_coords[u] = coords[0]

        # Use the last coordinate of the edge for node v
        if v not in node_coords:
            node_coords[v] = coords[-1]

    if not node_coords:
        raise ValueError("Cannot build KDTree: no node coordinates could be derived from edge geometries.")

    nodes = list(node_coords.keys())
    arr = np.array([node_coords[n] for n in nodes], dtype=float)

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"KDTree input has invalid shape: {arr.shape}")

    tree = cKDTree(arr)
    return nodes, arr, tree


def snap_point_to_graph_node(x, y, nodes, tree):
    dist, idx = tree.query([x, y], k=1)
    return nodes[idx], float(dist)


def build_preference_weight_function(steepness_factor=4.0, crime_factor=0.0, noise_factor=0.0):
    """
    Build a dynamic edge-weight function for personalized routing.
    Only steepness is active for now. Crime and noise are placeholder hooks so the API contract can remain stable when those datasets are added later.
    """

    # This function is passed directly into NetworkX so the route cost can change dynamically with the user's current preferences.
    def weight(u, v, data):
        length_m = float(data.get("length_m", 0.0))
        slope_pct = float(data.get("slope_pct", 0.0))
        type_factor = float(data.get("type_factor", 1.0))

        slope_penalty = 1.0 + float(steepness_factor) * max(slope_pct, 0.0) / 100.0

        # Placeholder multipliers for future crime / noise integration.
        crime_penalty = 1.0 + float(crime_factor) * float(data.get("crime_penalty", 0.0))
        noise_penalty = 1.0 + float(noise_factor) * float(data.get("noise_penalty", 0.0))

        return length_m * type_factor * slope_penalty * crime_penalty * noise_penalty

    return weight

# Route solving
def solve_route(G: nx.Graph, source_node, target_node, weight_field):
    """
    NetworkX Dijkstra shortest path
    """
    node_path = nx.shortest_path(G, source=source_node, target=target_node, weight=weight_field)

    edge_rows = []
    total_length = 0.0
    weighted_slope_sum = 0.0
    footpath_length = 0.0
    road_length = 0.0

    for u, v in zip(node_path[:-1], node_path[1:]):
        data = G[u][v]
        edge_type = data.get("edge_type", "road")

        edge_rows.append({
            "u": u,
            "v": v,
            "length_m": data["length_m"],
            "slope_pct": data["slope_pct"],
            "edge_type": edge_type,
            "display_type": data.get("display_type", edge_type),
            "cost_shortest": data["cost_shortest"],
            "cost_easiest": data["cost_easiest"],
            "geometry": data["geometry"]
        })

        total_length += data["length_m"]
        weighted_slope_sum += data["slope_pct"] * data["length_m"]

        if edge_type == "footpath":
            footpath_length += data["length_m"]
        else:
            road_length += data["length_m"]

    avg_slope = weighted_slope_sum / total_length if total_length > 0 else 0.0

    route_gdf = gpd.GeoDataFrame(edge_rows, geometry="geometry", crs="EPSG:27700")

    stats = {
        "total_length_m": total_length,
        "average_slope_pct": avg_slope,
        "road_length_m": road_length,
        "footpath_length_m": footpath_length,
        "footpath_share": footpath_length / total_length if total_length > 0 else 0.0,
        "edge_count": len(edge_rows),
        "node_count": len(node_path)
    }

    return node_path, route_gdf, stats


# export
def export_route_geojson(route_gdf: gpd.GeoDataFrame, out_path: str):
    out = route_gdf.to_crs(4326)
    out.to_file(out_path, driver="GeoJSON")
    return out

# network classification
def split_full_network_by_fclass(full_gdf: gpd.GeoDataFrame):
    """
    Split a full network into roads and footpaths using the fclass field.
    """
    full_gdf = ensure_linestrings(full_gdf).copy()

    #Exclude types that should not currently be used for walking routing
    # cycleways can be excluded to avoid routes that feel too bike-oriented
    exclude_types = {"cycleway"}
    full_gdf = full_gdf[~full_gdf["fclass"].isin(exclude_types)].copy()

    footpath_types = {"footway", "path", "pedestrian", "steps"}

    footpaths_gdf = full_gdf[full_gdf["fclass"].isin(footpath_types)].copy()
    roads_gdf = full_gdf[~full_gdf["fclass"].isin(footpath_types)].copy()

    return roads_gdf.reset_index(drop=True), footpaths_gdf.reset_index(drop=True)


def build_combined_network_from_full(full_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Single-file input (for example network_full_with_slope.gpkg)
    -> split internally into roads / footpaths
    -> merge back together with an edge_type field.
    """
    roads_gdf, footpaths_gdf = split_full_network_by_fclass(full_gdf)

    roads_gdf = roads_gdf.copy()
    footpaths_gdf = footpaths_gdf.copy()

    roads_gdf["edge_type"] = "road"
    footpaths_gdf["edge_type"] = "footpath"

    combined = pd.concat([roads_gdf, footpaths_gdf], ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=full_gdf.crs)

# Main end-to-end workflow
def prepare_network_and_routes(
    full_gdf: gpd.GeoDataFrame,
    sample_start_xy,
    sample_end_xy,
    snap_tolerance=3.0,
    footpath_penalty=1.08,
    slope_factor=4,
    out_shortest="route_shortest.geojson",
    out_easiest="route_easiest.geojson",
    out_network="network_processed.geojson"
):
    """
    Main workflow:
    0) split and rebuild a single-file network (road + footpath)
    1) clean lines
    2) snap nearby endpoints
    3) split at intersections
    4) map attributes back
    5) build the graph
    6) keep only the largest connected component
    7) snap sample points to nodes in the main connected component
    8) run shortest / easiest routing
    9) export GeoJSON
    """

    print("Step 0: split full network and combine roads + footpaths...")
    combined_input = build_combined_network_from_full(full_gdf)

    print(f"Combined input edges: {len(combined_input)}")
    print(combined_input["edge_type"].value_counts(dropna=False))

    print("Step 1: ensure linestrings...")
    base = ensure_linestrings(combined_input)

    if "length_m" not in base.columns:
        base["length_m"] = base.geometry.length
    if "slope_pct" not in base.columns:
        base["slope_pct"] = 0.0
    if "edge_type" not in base.columns:
        base["edge_type"] = "road"

    base["length_m"] = pd.to_numeric(base["length_m"], errors="coerce")
    base["slope_pct"] = pd.to_numeric(base["slope_pct"], errors="coerce").fillna(0.0)
    base = base[base["length_m"].notna()].copy()

    print("Step 2: snap nearby endpoints...")
    snapped = snap_nearby_endpoints(base, tolerance=snap_tolerance)

    print("Step 3: split at intersections...")
    noded_segments = extract_noded_segments(snapped)

    print("Step 4: map attributes back to split segments...")
    processed = map_attributes_to_segments(
        original_gdf=snapped,
        seg_gdf=noded_segments,
        attr_cols=[c for c in snapped.columns if c != "geometry"]
    )

    processed = processed[processed.geometry.length > 0].copy()
    processed["length_m"] = processed.geometry.length

    if "edge_type" not in processed.columns:
        processed["edge_type"] = "road"

    print("Step 5: build graph...")
    G = build_graph_from_segments(
        processed,
        footpath_penalty=footpath_penalty,
        slope_factor=slope_factor
    )

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    components = list(nx.connected_components(G))
    components = sorted(components, key=len, reverse=True)
    print(f"Connected components: {len(components)}")
    print(f"Largest component size: {len(components[0]) if components else 0}")

    if not components:
        raise ValueError("Graph is empty, routing cannot be performed.")

    largest_cc = components[0]
    G_main = G.subgraph(largest_cc).copy()
    print(f"Main component graph: {G_main.number_of_nodes()} nodes, {G_main.number_of_edges()} edges")

    print("Step 6: build KDTree for nodes...")
    nodes, arr, tree = build_node_kdtree(G_main)

    print("Step 7: snap manual coordinates to nearest graph nodes...")
    start_node, start_dist = snap_point_to_graph_node(sample_start_xy[0], sample_start_xy[1], nodes, tree)
    end_node, end_dist = snap_point_to_graph_node(sample_end_xy[0], sample_end_xy[1], nodes, tree)

    print(f"Start node snapped, distance = {start_dist:.2f} m")
    print(f"End node snapped, distance = {end_dist:.2f} m")

    if start_dist > 100:
        print("Warning: the start point is far from the main network; the test point may not be ideal.")
    if end_dist > 100:
        print("Warning: the end point is far from the main network; the test point may not be ideal.")

    print("Step 8: shortest route...")
    shortest_nodes, shortest_route, shortest_stats = solve_route(
        G_main, start_node, end_node, weight_field="cost_shortest"
    )

    print("Step 9: easiest route...")
    easiest_nodes, easiest_route, easiest_stats = solve_route(
        G_main, start_node, end_node, weight_field="cost_easiest"
    )

    print("Step 10: export processed network and routes...")

    processed_4326 = processed.to_crs(4326)
    processed_4326.to_file(out_network, driver="GeoJSON")

    main_nodes_set = set(G_main.nodes)
    processed_main = processed[
        processed.geometry.apply(
            lambda geom: (
                round_coord(*list(geom.coords)[0]) in main_nodes_set and
                round_coord(*list(geom.coords)[-1]) in main_nodes_set
            )
        )
    ].copy()
    processed_main_4326 = processed_main.to_crs(4326)
    processed_main_path = out_network.replace(".geojson", "_main_component.geojson")
    processed_main_4326.to_file(processed_main_path, driver="GeoJSON")

    shortest_4326 = export_route_geojson(shortest_route, out_shortest)
    easiest_4326 = export_route_geojson(easiest_route, out_easiest)

    return {
        "processed_network": processed,
        "processed_network_main": processed_main,
        "graph": G,
        "graph_main": G_main,
        "start_node": start_node,
        "end_node": end_node,
        "start_snap_distance_m": start_dist,
        "end_snap_distance_m": end_dist,
        "shortest_route": shortest_route,
        "easiest_route": easiest_route,
        "shortest_stats": shortest_stats,
        "easiest_stats": easiest_stats,
        "processed_network_4326": processed_4326,
        "processed_network_main_4326": processed_main_4326,
        "shortest_route_4326": shortest_4326,
        "easiest_route_4326": easiest_4326,
        "processed_main_path": processed_main_path
    }


if __name__ == "__main__":
    import geopandas as gpd

    network_path = DATA_DIR / "network_full_with_slope.gpkg"
    network_layer = "network_full_with_slope"

    full_gdf = gpd.read_file(network_path, layer=network_layer)

    if full_gdf.crs is None:
        raise ValueError("The network file does not contain CRS information.")
    if full_gdf.crs.to_epsg() != 27700:
        full_gdf = full_gdf.to_crs(27700)


    sample_start_xy = (530000, 181000)
    sample_end_xy = (535000, 182000)

    result = prepare_network_and_routes(
        full_gdf=full_gdf,
        sample_start_xy=sample_start_xy,
        sample_end_xy=sample_end_xy,
        snap_tolerance=3.0,
        footpath_penalty=1.08,
        slope_factor=4.0,
        out_shortest=str(DATA_DIR / "route_shortest.geojson"),
        out_easiest=str(DATA_DIR / "route_easiest.geojson"),
        out_network=str(DATA_DIR / "network_processed.geojson")
    )

    print("\n=== Shortest route stats ===")
    print(result["shortest_stats"])

    print("\n=== Easiest route stats ===")
    print(result["easiest_stats"])