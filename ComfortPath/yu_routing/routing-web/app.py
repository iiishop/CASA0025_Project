"""Flask backend for the routing prototype."""

import pickle
from pathlib import Path

import requests

from flask import Flask, render_template, request, jsonify
from pyproj import Transformer

from routing import (
    build_node_kdtree,
    build_preference_weight_function,
    snap_point_to_graph_node,
    solve_route,
)


# Flask app
app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


GRAPH_CACHE_PATH = DATA_DIR / "main_graph.pkl"

print("Loading cached graph...")
with open(GRAPH_CACHE_PATH, "rb") as f:
    G_main = pickle.load(f)

print(f"Graph loaded: {G_main.number_of_nodes()} nodes, {G_main.number_of_edges()} edges")

print("Building KDTree...")
nodes, arr, tree = build_node_kdtree(G_main)
print("KDTree ready.")

# EPSG:4326 -> EPSG:27700
transformer_to_27700 = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)


NOMINATIM_BASE = "https://nominatim.openstreetmap.org"
NOMINATIM_HEADERS = {
    "User-Agent": "CASA0025-Routing-Prototype"
}

# Maps slider level (1–3) to MCDM weight; level 2 is the default balanced setting
PREFERENCE_WEIGHT_MAP = {1: 0.2, 2: 1.0, 3: 4.0}

## functions
def route_gdf_to_geojson(route_gdf):
    """
    GeoDataFrame(EPSG:27700) -> GeoJSON dict(EPSG:4326)
    """
    route_4326 = route_gdf.to_crs(4326)
    return route_4326.__geo_interface__


def format_stats(stats: dict):
    avg_walking_effort = round(stats.get("average_walking_effort", 0.0), 4)
    return {
        "length_m": round(stats["total_length_m"], 2),
        "avg_walking_effort": avg_walking_effort,
        "avg_slope_score": avg_walking_effort,
        "road_length_m": round(stats["road_length_m"], 2),
        "footpath_length_m": round(stats["footpath_length_m"], 2),
        "footpath_share": round(stats["footpath_share"], 4),
        "edge_count": stats["edge_count"],
        "node_count": stats["node_count"],
    }


def nominatim_get(path: str, params: dict):
    """
    Proxy Nominatim requests through Flask so the frontend does not have to manage headers, CORS, or rate-limit related details directly.
    """
    response = requests.get(
        f"{NOMINATIM_BASE}{path}",
        params=params,
        headers=NOMINATIM_HEADERS,
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def compute_routes(origin, destination, preferences=None):
    """
    Compute shortest and personalised routes for one origin & destination pair.

    origin = {"lat": ..., "lng": ...}
    destination = {"lat": ..., "lng": ...}
    """
    preferences = preferences or {}

    walk_level = int(preferences.get("walk", 2))
    walk_level = max(1, min(3, walk_level))
    walk_factor = PREFERENCE_WEIGHT_MAP[walk_level]
    safety_factor = float(preferences.get("safety", 0.0))
    activity_factor = float(preferences.get("activity", 0.0))
    shade_shelter_factor = float(preferences.get("shade_shelter", 0.0))
    air_factor = float(preferences.get("air", 0.0))
    noise_factor = float(preferences.get("noise", 0.0))

    origin_lng = float(origin["lng"])
    origin_lat = float(origin["lat"])
    destination_lng = float(destination["lng"])
    destination_lat = float(destination["lat"])

    origin_x, origin_y = transformer_to_27700.transform(origin_lng, origin_lat)
    destination_x, destination_y = transformer_to_27700.transform(destination_lng, destination_lat)

    start_node, start_dist = snap_point_to_graph_node(origin_x, origin_y, nodes, tree)
    end_node, end_dist = snap_point_to_graph_node(destination_x, destination_y, nodes, tree)

    if start_dist > 300:
        raise ValueError(f"Origin is too far from the routed network ({start_dist:.1f} m).")

    if end_dist > 300:
        raise ValueError(f"Destination is too far from the routed network ({end_dist:.1f} m).")

    shortest_nodes, shortest_route_gdf, shortest_stats = solve_route(
        G_main, start_node, end_node, weight_field="cost_shortest"
    )

    # Personalised route uses a dynamic weight function built from precomputed road-segment scores, 
    # which keeps the web app fast at runtime.
    preference_weight = build_preference_weight_function(
        length_factor=1.0,
        walk_factor=walk_factor,
        safety_factor=safety_factor,
        activity_factor=activity_factor,
        shade_shelter_factor=shade_shelter_factor,
        air_factor=air_factor,
        noise_factor=noise_factor,
    )

    easiest_nodes, easiest_route_gdf, easiest_stats = solve_route(
        G_main, start_node, end_node, weight_field=preference_weight
    )

    result = {
        "shortest": {
            "geojson": route_gdf_to_geojson(shortest_route_gdf),
            "stats": format_stats(shortest_stats)
        },
        "easiest": {
            "geojson": route_gdf_to_geojson(easiest_route_gdf),
            "stats": format_stats(easiest_stats)
        },
        "debug": {
            "start_snap_distance_m": round(start_dist, 2),
            "end_snap_distance_m": round(end_dist, 2),
            "start_node": start_node,
            "end_node": end_node
        },
        "preferences": {
            "walk": walk_level,
            "walk_factor": walk_factor,
            "safety": safety_factor,
            "activity": activity_factor,
            "shade_shelter": shade_shelter_factor,
            "air": air_factor,
            "noise": noise_factor,
            "steepness": walk_level,
            "steepness_factor": walk_factor,
        }
    }

    return result

## Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/geocode", methods=["GET"])
def geocode():
    query = (request.args.get("q") or "").strip()

    if not query:
        return jsonify({"error": "Missing query parameter: q"}), 400

    try:
        results = nominatim_get(
            "/search",
            {
                "q": query,
                "format": "jsonv2",
                "limit": 5,
                "addressdetails": 1,
                "countrycodes": "gb",
            },
        )

        cleaned_results = [
            {
                "display_name": item.get("display_name", query),
                "lat": float(item["lat"]),
                "lng": float(item["lon"]),
            }
            for item in results
            if item.get("lat") and item.get("lon")
        ]

        return jsonify({"results": cleaned_results})

    except requests.RequestException as e:
        return jsonify({"error": f"Geocoding request failed: {str(e)}"}), 502


@app.route("/reverse_geocode", methods=["GET"])
def reverse_geocode():
    lat = request.args.get("lat")
    lng = request.args.get("lng")

    if lat is None or lng is None:
        return jsonify({"error": "Missing lat or lng"}), 400

    try:
        result = nominatim_get(
            "/reverse",
            {
                "lat": lat,
                "lon": lng,
                "format": "jsonv2",
                "zoom": 18,
                "addressdetails": 1,
            },
        )

        return jsonify(
            {
                "display_name": result.get("display_name", "Selected location"),
                "lat": float(result.get("lat", lat)),
                "lng": float(result.get("lon", lng)),
            }
        )

    except requests.RequestException as e:
        return jsonify({"error": f"Reverse geocoding request failed: {str(e)}"}), 502


@app.route("/route", methods=["POST"])
def route():
    """Main API endpoint for frontend route requests."""
    data = request.get_json()

    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    origin = data.get("origin")
    destination = data.get("destination")
    preferences = data.get("preferences", {})

    if not origin or not destination:
        return jsonify({"error": "Missing origin or destination"}), 400

    try:
        result = compute_routes(origin, destination, preferences)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)