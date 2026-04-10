from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import networkx as nx

from .air_quality_source import AirQualityRouteSource, RouteAirWeights
from .laei_service import AirQualityWeights, LAEIService
from .osm_service import OSMWalkService, RouteRequest


app = FastAPI(title="ComfortPath Air Quality API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


LAEI_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "laei"
OSM_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "osm"
laei_service = LAEIService(LAEI_DATA_DIR)
osm_service = OSMWalkService(
    pbf_path=OSM_DATA_DIR / "London.osm.pbf",
    cache_path=OSM_DATA_DIR / "london_walk_graph.pkl",
)
air_route_source = AirQualityRouteSource(laei_service)
startup_errors: dict[str, str] = {}
TILE_CACHE_CONTROL = "public, max-age=600"


def tile_png_response(content: bytes) -> Response:
    return Response(
        content=content,
        media_type="image/png",
        headers={"Cache-Control": TILE_CACHE_CONTROL},
    )


@app.on_event("startup")
def startup_event() -> None:
    startup_errors.clear()
    try:
        laei_service.load()
    except Exception as exc:
        startup_errors["laei"] = str(exc)


def ensure_ready(*services: str) -> None:
    targets = services or tuple(startup_errors.keys())
    for service in targets:
        err = startup_errors.get(service)
        if err:
            raise RuntimeError(f"{service} unavailable: {err}")


@app.get("/health")
def health() -> dict[str, str]:
    osm_state = osm_service.load_state()
    return {
        "status": "degraded" if startup_errors else "ok",
        "laei": "not_loaded" if "laei" in startup_errors else "loaded",
        "osm": "loaded" if osm_state["loaded"] else "lazy_not_loaded",
    }


@app.get("/factors")
def factors() -> dict:
    return {
        "factors": [
            {
                "id": "air_quality",
                "label": "Air Quality",
                "default_weight": 1.0,
                "range": [0.0, 1.0],
                "unit": "score_0_100",
            }
        ],
        "weights_schema": {
            "air_quality": {
                "components": ["no2", "pm25", "pm10"],
                "default": {"no2": 0.4, "pm25": 0.35, "pm10": 0.25},
            }
        },
    }


@app.get("/meta/air-quality/latest")
def latest_air_quality_meta() -> dict:
    try:
        ensure_ready("laei")
        return laei_service.metadata()
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch metadata: {exc}"
        ) from exc


@app.get("/tiles/air-quality")
def air_quality_tiles(
    request: Request,
    no2_weight: float = Query(default=0.4, ge=0.0, le=1.0),
    pm25_weight: float = Query(default=0.35, ge=0.0, le=1.0),
    pm10_weight: float = Query(default=0.25, ge=0.0, le=1.0),
) -> dict:
    try:
        ensure_ready("laei")
        base = str(request.base_url).rstrip("/")
        tile_url = (
            f"{base}/tiles/air-quality/{{z}}/{{x}}/{{y}}.png"
            f"?no2_weight={no2_weight}&pm25_weight={pm25_weight}&pm10_weight={pm10_weight}"
        )
        return {
            "layer": "air_quality",
            "tile_url": tile_url,
            "visualization": {
                "min": 0,
                "max": 100,
                "palette": ["2c7bb6", "a7d7f0", "fef3c7", "f97316", "b91c1c"],
            },
            "weights": {"no2": no2_weight, "pm25": pm25_weight, "pm10": pm10_weight},
            "note": "LAEI 2022 static raster tile endpoint.",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to build tile URL: {exc}"
        ) from exc


@app.get("/tiles/air-quality/{z}/{x}/{y}.png")
def air_quality_tile_png(
    z: int,
    x: int,
    y: int,
    no2_weight: float = Query(default=0.4, ge=0.0, le=1.0),
    pm25_weight: float = Query(default=0.35, ge=0.0, le=1.0),
    pm10_weight: float = Query(default=0.25, ge=0.0, le=1.0),
) -> Response:
    try:
        ensure_ready("laei")
        png = laei_service.render_tile_png(
            z=z,
            x=x,
            y=y,
            weights=AirQualityWeights(
                no2=no2_weight, pm25=pm25_weight, pm10=pm10_weight
            ),
        )
        return tile_png_response(png)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to render tile: {exc}"
        ) from exc


@app.get("/vectors/osm/roads")
def osm_roads_vector(
    lon_min: float = Query(..., ge=-180.0, le=180.0),
    lat_min: float = Query(..., ge=-90.0, le=90.0),
    lon_max: float = Query(..., ge=-180.0, le=180.0),
    lat_max: float = Query(..., ge=-90.0, le=90.0),
    zoom: int = Query(default=12, ge=0, le=22),
) -> dict:
    try:
        osm_service.ensure_loaded()
        if lon_min >= lon_max or lat_min >= lat_max:
            raise HTTPException(status_code=400, detail="Invalid bounding box")

        return osm_service.roads_geojson(
            lon_min=lon_min,
            lat_min=lat_min,
            lon_max=lon_max,
            lat_max=lat_max,
            zoom=zoom,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to render OSM vector roads: {exc}"
        ) from exc


@app.get("/meta/osm/walking")
def osm_walking_meta() -> dict:
    try:
        osm_service.ensure_loaded()
        return osm_service.metadata()
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch OSM metadata: {exc}"
        ) from exc


@app.get("/route/walking")
def walking_route(
    start_lon: float = Query(..., ge=-180.0, le=180.0),
    start_lat: float = Query(..., ge=-90.0, le=90.0),
    end_lon: float = Query(..., ge=-180.0, le=180.0),
    end_lat: float = Query(..., ge=-90.0, le=90.0),
    distance_weight: float = Query(default=1.0, ge=0.0, le=10.0),
    air_quality_weight: float = Query(default=1.0, ge=0.0, le=10.0),
    no2_weight: float = Query(default=0.4, ge=0.0, le=1.0),
    pm25_weight: float = Query(default=0.35, ge=0.0, le=1.0),
    pm10_weight: float = Query(default=0.25, ge=0.0, le=1.0),
) -> dict:
    try:
        ensure_ready("laei")
        osm_service.ensure_loaded()
        air_weights = RouteAirWeights(
            air_quality=air_quality_weight,
            no2=no2_weight,
            pm25=pm25_weight,
            pm10=pm10_weight,
        )

        def edge_cost_fn(_: int, __: int, data: dict) -> float:
            exposure = air_route_source.edge_exposure_score(data, air_weights)
            return exposure * data["length_m"] * air_weights.air_quality

        result = osm_service.route(
            RouteRequest(
                start_lon=start_lon,
                start_lat=start_lat,
                end_lon=end_lon,
                end_lat=end_lat,
            ),
            distance_weight=distance_weight,
            edge_cost_fn=edge_cost_fn,
        )

        total_air_exposure = 0.0
        for edge in result["edges"]:
            total_air_exposure += (
                air_route_source.edge_exposure_score(edge, air_weights)
                * edge["length_m"]
            )

        coordinates = result["coordinates"]
        return {
            "profile": "walking",
            "summary": {
                "distance_m": result["distance_m"],
                "air_exposure_distance_weighted": total_air_exposure,
                "node_count": len(coordinates),
            },
            "weights": {
                "distance": distance_weight,
                "air_quality": air_quality_weight,
                "pollutants": {
                    "no2": no2_weight,
                    "pm25": pm25_weight,
                    "pm10": pm10_weight,
                },
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates,
            },
        }
    except nx.NetworkXNoPath as exc:
        raise HTTPException(status_code=404, detail="No walking path found") from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to build walking route: {exc}"
        ) from exc
