from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from .laei_service import AirQualityWeights, LAEIService


app = FastAPI(title="ComfortPath Air Quality API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


LAEI_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "laei"
laei_service = LAEIService(LAEI_DATA_DIR)
startup_error: str | None = None


@app.on_event("startup")
def startup_event() -> None:
    global startup_error
    try:
        laei_service.load()
        startup_error = None
    except Exception as exc:
        startup_error = str(exc)


def ensure_ready() -> None:
    if startup_error:
        raise RuntimeError(startup_error)


@app.get("/health")
def health() -> dict[str, str]:
    if startup_error:
        return {"status": "degraded", "laei": "not_loaded"}
    return {"status": "ok", "laei": "loaded"}


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
        ensure_ready()
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
        ensure_ready()
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
        ensure_ready()
        png = laei_service.render_tile_png(
            z=z,
            x=x,
            y=y,
            weights=AirQualityWeights(
                no2=no2_weight, pm25=pm25_weight, pm10=pm10_weight
            ),
        )
        return Response(content=png, media_type="image/png")
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to render tile: {exc}"
        ) from exc
