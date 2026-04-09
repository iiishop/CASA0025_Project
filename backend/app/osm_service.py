from __future__ import annotations

import math
import pickle
import threading
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable

import networkx as nx
import numpy as np
import osmium
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm


ALLOWED_HIGHWAYS = {
    "footway",
    "path",
    "pedestrian",
    "living_street",
    "track",
    "service",
    "residential",
    "unclassified",
    "tertiary",
    "tertiary_link",
    "secondary",
    "secondary_link",
    "primary",
    "primary_link",
    "steps",
    "corridor",
    "cycleway",
}

NO_FOOT_VALUES = {"no", "private"}
ONEWAY_TRUE = {"yes", "1", "true"}
ONEWAY_REVERSE = {"-1"}


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@dataclass(frozen=True)
class RouteRequest:
    start_lon: float
    start_lat: float
    end_lon: float
    end_lat: float


class _WalkingGraphBuilder(osmium.SimpleHandler):
    def __init__(self, progress: tqdm | None = None):
        super().__init__()
        self.graph = nx.DiGraph()
        self.node_coords: dict[int, tuple[float, float]] = {}
        self.edge_count = 0
        self.progress = progress
        self._way_tick = 0

    def _is_walkable(self, tags: dict[str, str]) -> tuple[bool, str]:
        highway = tags.get("highway", "")
        if highway not in ALLOWED_HIGHWAYS:
            return False, "both"

        if tags.get("foot", "").lower() in NO_FOOT_VALUES:
            return False, "both"
        if tags.get("access", "").lower() in NO_FOOT_VALUES:
            return False, "both"

        oneway = tags.get("oneway", "").lower()
        oneway_foot = tags.get("oneway:foot", "").lower()

        if oneway_foot == "no":
            return True, "both"
        if oneway in ONEWAY_REVERSE:
            return True, "reverse"
        if oneway in ONEWAY_TRUE or tags.get("junction", "") == "roundabout":
            return True, "forward"
        return True, "both"

    def _add_edge(self, u: int, v: int, data: dict) -> None:
        if self.graph.has_edge(u, v):
            existing = self.graph[u][v]
            if data["length_m"] < existing["length_m"]:
                self.graph[u][v] = data
            return
        self.graph.add_edge(u, v, **data)

    def way(self, w: osmium.osm.Way) -> None:
        self._way_tick += 1
        if self.progress is not None and self._way_tick % 1000 == 0:
            self.progress.update(1000)

        tags = {k: v for k, v in w.tags}
        walkable, direction = self._is_walkable(tags)
        if not walkable:
            return

        refs: list[int] = []
        for n in w.nodes:
            if not n.location.valid():
                continue
            nid = int(n.ref)
            self.node_coords[nid] = (n.location.lon, n.location.lat)
            refs.append(nid)

        if len(refs) < 2:
            return

        for i in range(len(refs) - 1):
            a = refs[i]
            b = refs[i + 1]
            if a == b:
                continue

            lon1, lat1 = self.node_coords[a]
            lon2, lat2 = self.node_coords[b]
            length_m = haversine_m(lon1, lat1, lon2, lat2)
            if length_m <= 0:
                continue

            mid_lon = (lon1 + lon2) * 0.5
            mid_lat = (lat1 + lat2) * 0.5
            base_data = {
                "edge_id": self.edge_count,
                "length_m": float(length_m),
                "highway": tags.get("highway"),
                "surface": tags.get("surface"),
                "way_id": int(w.id),
                "name": tags.get("name"),
                "mid_lon": float(mid_lon),
                "mid_lat": float(mid_lat),
            }
            self.edge_count += 1

            if direction in {"both", "forward"}:
                self._add_edge(a, b, base_data)
            if direction in {"both", "reverse"}:
                reverse_data = dict(base_data)
                reverse_data["edge_id"] = self.edge_count
                self.edge_count += 1
                self._add_edge(b, a, reverse_data)


class OSMWalkService:
    def __init__(self, pbf_path: Path, cache_path: Path):
        self.pbf_path = pbf_path
        self.cache_path = cache_path
        self.graph = nx.DiGraph()
        self.node_ids: np.ndarray | None = None
        self.node_lons: np.ndarray | None = None
        self.node_lats: np.ndarray | None = None
        self.node_index: dict[int, int] = {}
        self.edge_segments: list[tuple[float, float, float, float]] = []
        self.edge_bounds: list[tuple[float, float, float, float]] = []
        self.edge_spatial_index: dict[tuple[int, int], list[int]] = {}
        self.index_cell_deg = 0.01
        self._loaded = False
        self._load_error: str | None = None
        self._load_lock = threading.Lock()
        self._mask_cache: OrderedDict[tuple[int, int, int, int, int], bytes] = (
            OrderedDict()
        )
        self._mask_cache_limit = 4096

    def _do_load(self) -> None:
        if self.cache_path.exists():
            self._load_cache()
            self._loaded = True
            return

        if not self.pbf_path.exists():
            raise RuntimeError(f"Missing OSM PBF file: {self.pbf_path}")

        progress = tqdm(desc="OSM ways", unit="way", dynamic_ncols=True)
        builder = _WalkingGraphBuilder(progress=progress)
        builder.apply_file(str(self.pbf_path), locations=True)
        if builder.progress is not None:
            remainder = builder._way_tick % 1000
            if remainder:
                builder.progress.update(remainder)
            builder.progress.close()

        self.graph = builder.graph

        for nid, (lon, lat) in builder.node_coords.items():
            if nid in self.graph:
                self.graph.nodes[nid]["lon"] = float(lon)
                self.graph.nodes[nid]["lat"] = float(lat)

        self._build_index()
        self._save_cache()
        self._loaded = True

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            try:
                self._do_load()
                self._load_error = None
            except Exception as exc:
                self._load_error = str(exc)
                raise

    def _build_index(self) -> None:
        nodes = list(self.graph.nodes(data=True))
        self.node_ids = np.array([nid for nid, _ in nodes], dtype=np.int64)
        self.node_lons = np.array(
            [attrs["lon"] for _, attrs in nodes], dtype=np.float64
        )
        self.node_lats = np.array(
            [attrs["lat"] for _, attrs in nodes], dtype=np.float64
        )
        self.node_index = {
            int(nid): idx for idx, nid in enumerate(self.node_ids.tolist())
        }
        self._build_edge_index()

    def _build_edge_index(self) -> None:
        self.edge_segments = []
        self.edge_bounds = []
        self.edge_spatial_index = {}

        for u, v in self.graph.edges():
            u_node = self.graph.nodes[u]
            v_node = self.graph.nodes[v]
            lon1 = float(u_node["lon"])
            lat1 = float(u_node["lat"])
            lon2 = float(v_node["lon"])
            lat2 = float(v_node["lat"])

            min_lon = min(lon1, lon2)
            max_lon = max(lon1, lon2)
            min_lat = min(lat1, lat2)
            max_lat = max(lat1, lat2)

            seg_id = len(self.edge_segments)
            self.edge_segments.append((lon1, lat1, lon2, lat2))
            self.edge_bounds.append((min_lon, min_lat, max_lon, max_lat))

            x0 = math.floor(min_lon / self.index_cell_deg)
            x1 = math.floor(max_lon / self.index_cell_deg)
            y0 = math.floor(min_lat / self.index_cell_deg)
            y1 = math.floor(max_lat / self.index_cell_deg)

            for xi in range(x0, x1 + 1):
                for yi in range(y0, y1 + 1):
                    key = (xi, yi)
                    self.edge_spatial_index.setdefault(key, []).append(seg_id)

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("wb") as f:
            pickle.dump(self.graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_cache(self) -> None:
        with self.cache_path.open("rb") as f:
            self.graph = pickle.load(f)
        self._build_index()

    @staticmethod
    def tile_bounds_wgs84(z: int, x: int, y: int) -> tuple[float, float, float, float]:
        n = 2**z
        lon_left = x / n * 360.0 - 180.0
        lon_right = (x + 1) / n * 360.0 - 180.0

        def lat_from_tile(ty: int) -> float:
            val = math.pi * (1 - 2 * ty / n)
            return math.degrees(math.atan(math.sinh(val)))

        lat_top = float(lat_from_tile(y))
        lat_bottom = float(lat_from_tile(y + 1))
        return lon_left, lat_bottom, lon_right, lat_top

    def _candidate_segments(
        self,
        lon_min: float,
        lat_min: float,
        lon_max: float,
        lat_max: float,
        margin_deg: float,
    ) -> list[int]:
        x0 = math.floor((lon_min - margin_deg) / self.index_cell_deg)
        x1 = math.floor((lon_max + margin_deg) / self.index_cell_deg)
        y0 = math.floor((lat_min - margin_deg) / self.index_cell_deg)
        y1 = math.floor((lat_max + margin_deg) / self.index_cell_deg)

        candidates: set[int] = set()
        for xi in range(x0, x1 + 1):
            for yi in range(y0, y1 + 1):
                for seg_id in self.edge_spatial_index.get((xi, yi), []):
                    candidates.add(seg_id)
        return list(candidates)

    def road_mask_png(
        self,
        z: int,
        x: int,
        y: int,
        buffer_px: int = 3,
        apply_threshold: bool = True,
    ) -> bytes:
        cache_key = (z, x, y, buffer_px, int(apply_threshold))
        cached = self._mask_cache.get(cache_key)
        if cached is not None:
            self._mask_cache.move_to_end(cache_key)
            return cached

        lon_min, lat_min, lon_max, lat_max = self.tile_bounds_wgs84(z, x, y)

        margin_lon = (lon_max - lon_min) * max(1, buffer_px) / 256.0
        margin_lat = (lat_max - lat_min) * max(1, buffer_px) / 256.0
        margin_deg = max(margin_lon, margin_lat)
        candidates = self._candidate_segments(
            lon_min, lat_min, lon_max, lat_max, margin_deg
        )

        mask = Image.new("L", (256, 256), 0)
        draw = ImageDraw.Draw(mask)
        if z <= 10:
            zoom_scale = 0.22
        elif z <= 12:
            zoom_scale = 0.4
        elif z <= 14:
            zoom_scale = 0.65
        else:
            zoom_scale = 1.0

        width_px = max(1, int(round((buffer_px * 1.1) * zoom_scale)))

        lon_span = max(lon_max - lon_min, 1e-9)
        lat_span = max(lat_max - lat_min, 1e-9)

        for seg_id in candidates:
            b_lon_min, b_lat_min, b_lon_max, b_lat_max = self.edge_bounds[seg_id]
            if (
                b_lon_max < lon_min - margin_deg
                or b_lon_min > lon_max + margin_deg
                or b_lat_max < lat_min - margin_deg
                or b_lat_min > lat_max + margin_deg
            ):
                continue

            lon1, lat1, lon2, lat2 = self.edge_segments[seg_id]
            x1p = (lon1 - lon_min) / lon_span * 256.0
            y1p = (lat_max - lat1) / lat_span * 256.0
            x2p = (lon2 - lon_min) / lon_span * 256.0
            y2p = (lat_max - lat2) / lat_span * 256.0
            draw.line([(x1p, y1p), (x2p, y2p)], fill=255, width=width_px)

        if buffer_px > 1:
            blur_radius = max(0.2, buffer_px * 0.35 * zoom_scale)
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        if apply_threshold:
            if z <= 10:
                threshold = 52
            elif z <= 12:
                threshold = 24
            else:
                threshold = 8

            mask = mask.point(lambda p: 0 if p < threshold else p)

        buf = BytesIO()
        mask.save(buf, format="PNG")
        value = buf.getvalue()
        self._mask_cache[cache_key] = value
        if len(self._mask_cache) > self._mask_cache_limit:
            self._mask_cache.popitem(last=False)
        return value

    def _nearest_node(self, lon: float, lat: float) -> int:
        if self.node_lons is None or self.node_lats is None or self.node_ids is None:
            raise RuntimeError("OSM graph index not loaded")

        scale = math.cos(math.radians(lat))
        dx = (self.node_lons - lon) * scale
        dy = self.node_lats - lat
        idx = int(np.argmin(dx * dx + dy * dy))
        return int(self.node_ids[idx])

    def route(
        self,
        req: RouteRequest,
        distance_weight: float,
        edge_cost_fn: Callable[[int, int, dict], float] | None = None,
    ) -> dict:
        start = self._nearest_node(req.start_lon, req.start_lat)
        end = self._nearest_node(req.end_lon, req.end_lat)

        def weight_fn(u: int, v: int, data: dict) -> float:
            base = data["length_m"] * distance_weight
            if edge_cost_fn is None:
                return base
            return base + max(edge_cost_fn(u, v, data), 0.0)

        end_lon = self.graph.nodes[end]["lon"]
        end_lat = self.graph.nodes[end]["lat"]

        def heuristic(nid: int, _: int) -> float:
            lon = self.graph.nodes[nid]["lon"]
            lat = self.graph.nodes[nid]["lat"]
            return haversine_m(lon, lat, end_lon, end_lat) * distance_weight

        path = nx.astar_path(
            self.graph, start, end, heuristic=heuristic, weight=weight_fn
        )

        coords = [
            [self.graph.nodes[n]["lon"], self.graph.nodes[n]["lat"]] for n in path
        ]

        edges: list[dict] = []
        total_distance = 0.0
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            data = self.graph[u][v]
            edges.append(data)
            total_distance += float(data["length_m"])

        return {
            "start_node": start,
            "end_node": end,
            "coordinates": coords,
            "edges": edges,
            "distance_m": total_distance,
        }

    def metadata(self) -> dict:
        return {
            "dataset": "OpenStreetMap",
            "extract": str(self.pbf_path.name),
            "network": "walking",
            "loaded": self._loaded,
            "load_error": self._load_error,
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
        }

    def load_state(self) -> dict:
        return {
            "loaded": self._loaded,
            "load_error": self._load_error,
        }
