from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
import rasterio
import requests
from rasterio.merge import merge as merge_rasters

from app.config import get_settings
from app.gee_service import get_london_geometry, init_ee


OUT_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "ndvi" / "gee_ndvi_london_2023.tif"
)
CHUNK_DIR = Path(__file__).resolve().parents[1] / "data" / "ndvi" / "_ndvi_chunks"
CHUNK_SIZE_DEG = 0.2


def extract_bbox(geometry: dict) -> tuple[float, float, float, float]:
    coords = geometry.get("coordinates", [])
    xs: list[float] = []
    ys: list[float] = []

    def walk(node) -> None:
        if (
            isinstance(node, (list, tuple))
            and len(node) >= 2
            and isinstance(node[0], (int, float))
        ):
            xs.append(float(node[0]))
            ys.append(float(node[1]))
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(coords)
    if not xs or not ys:
        raise RuntimeError("Unable to derive London bbox from geometry")
    return min(xs), min(ys), max(xs), max(ys)


def iter_chunks(
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
) -> list[tuple[float, float, float, float]]:
    chunks: list[tuple[float, float, float, float]] = []
    lon = lon_min
    while lon < lon_max - 1e-9:
        next_lon = min(lon + CHUNK_SIZE_DEG, lon_max)
        lat = lat_min
        while lat < lat_max - 1e-9:
            next_lat = min(lat + CHUNK_SIZE_DEG, lat_max)
            chunks.append((lon, lat, next_lon, next_lat))
            lat = next_lat
        lon = next_lon
    return chunks


def main() -> None:
    settings = get_settings()
    init_ee(settings)
    london = get_london_geometry(settings)

    response = requests.get(settings.london_geojson_url, timeout=60)
    response.raise_for_status()
    features = response.json().get("features", [])
    if not features:
        raise RuntimeError("No London feature found in boundary source")
    lon_min, lat_min, lon_max, lat_max = extract_bbox(features[0]["geometry"])

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(london)
        .filterDate("2023-01-01", "2023-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 35))
    )

    composite = s2.median()
    ndvi = (
        composite.normalizedDifference(["B8", "B4"])
        .rename("ndvi")
        .focal_mean(radius=40, units="meters")
        .clip(london)
        .clamp(-1, 1)
    )

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    for old in CHUNK_DIR.glob("*.tif"):
        old.unlink()

    chunk_paths: list[Path] = []
    chunks = iter_chunks(lon_min, lat_min, lon_max, lat_max)
    print(f"Exporting {len(chunks)} NDVI chunks at 20m...")
    for idx, (c_lon_min, c_lat_min, c_lon_max, c_lat_max) in enumerate(chunks):
        region = ee.Geometry.Rectangle(
            [c_lon_min, c_lat_min, c_lon_max, c_lat_max],
            proj="EPSG:4326",
            geodesic=False,
        )
        try:
            url = ndvi.getDownloadURL(
                {
                    "region": region,
                    "scale": 20,
                    "crs": "EPSG:4326",
                    "format": "GEO_TIFF",
                }
            )
            chunk_resp = requests.get(url, timeout=300)
            chunk_resp.raise_for_status()
        except Exception as exc:
            message = str(exc)
            if "No valid (non-null) pixels" in message:
                continue
            raise RuntimeError(f"Failed NDVI chunk export at {idx}: {message}") from exc

        out_path = CHUNK_DIR / f"chunk_{idx:04d}.tif"
        out_path.write_bytes(chunk_resp.content)
        chunk_paths.append(out_path)

    if not chunk_paths:
        raise RuntimeError("No NDVI chunks exported from GEE")

    datasets = [rasterio.open(path) for path in chunk_paths]
    try:
        mosaic, transform = merge_rasters(datasets, nodata=float("nan"))
        meta = datasets[0].meta.copy()
        meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "count": 1,
                "dtype": "float32",
                "nodata": float("nan"),
                "compress": "deflate",
            }
        )
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(OUT_PATH, "w", **meta) as dst:
            dst.write(mosaic[0].astype("float32"), 1)
    finally:
        for ds in datasets:
            ds.close()

    print(f"Saved NDVI raster: {OUT_PATH}")


if __name__ == "__main__":
    main()
