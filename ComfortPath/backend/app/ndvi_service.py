from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject


@dataclass(frozen=True)
class NDVIMetadata:
    source: str
    year: int
    resolution: str
    p5: float
    p95: float
    path: str


class NDVIService:
    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.raster_path = data_root / "gee_ndvi_london_2023.tif"
        self.array: np.ndarray | None = None
        self.src_transform = None
        self.src_crs = None
        self.src_nodata: float | None = None
        self.percentiles: tuple[float, float] = (0.0, 1.0)
        self.selected_scale_m = 20

    def load(self) -> None:
        if not self.raster_path.exists():
            raise RuntimeError(
                f"Missing NDVI raster: {self.raster_path}. Run backend/scripts/download_ndvi_from_gee.py first."
            )

        with rasterio.open(self.raster_path) as ds:
            arr = ds.read(1).astype(np.float32)
            nodata = ds.nodata
            if nodata is not None:
                arr = np.where(arr == nodata, np.nan, arr)
            arr = np.where(np.isfinite(arr), np.clip(arr, -1.0, 1.0), np.nan)

            self.array = arr
            self.src_transform = ds.transform
            self.src_crs = ds.crs
            self.src_nodata = nodata

            res_x = abs(float(ds.transform.a))
            self.selected_scale_m = max(1, int(round(res_x * 111_320)))

        valid = np.isfinite(self.array)
        if not np.any(valid):
            raise RuntimeError(
                "No valid NDVI pixels available after loading local raster"
            )

        values = self.array[valid]
        p5, p95 = np.percentile(values, [5, 95])
        if p95 <= p5:
            p5 = float(np.min(values))
            p95 = float(np.max(values))
        self.percentiles = (float(p5), float(p95))

    @staticmethod
    def tile_bounds_wgs84(z: int, x: int, y: int) -> tuple[float, float, float, float]:
        n = 2**z
        lon_left = x / n * 360.0 - 180.0
        lon_right = (x + 1) / n * 360.0 - 180.0

        def lat_from_tile(ty: int) -> float:
            val = np.pi * (1 - 2 * ty / n)
            return np.degrees(np.arctan(np.sinh(val)))

        lat_top = float(lat_from_tile(y))
        lat_bottom = float(lat_from_tile(y + 1))
        return lon_left, lat_bottom, lon_right, lat_top

    def read_tile(self, z: int, x: int, y: int) -> np.ndarray:
        if self.array is None or self.src_transform is None or self.src_crs is None:
            raise RuntimeError("NDVI service not loaded")

        lon_min, lat_min, lon_max, lat_max = self.tile_bounds_wgs84(z, x, y)
        data = np.full((256, 256), np.nan, dtype=np.float32)
        dst_transform = from_bounds(lon_min, lat_min, lon_max, lat_max, 256, 256)

        reproject(
            source=self.array,
            destination=data,
            src_transform=self.src_transform,
            src_crs=self.src_crs,
            src_nodata=np.nan,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
            init_dest_nodata=True,
            num_threads=2,
        )
        return data

    def normalize(self, arr: np.ndarray) -> np.ndarray:
        p5, p95 = self.percentiles
        denom = max(p95 - p5, 1e-6)
        return np.clip((arr - p5) / denom, 0.0, 1.0)

    def metadata(self) -> dict:
        p5, p95 = self.percentiles
        return NDVIMetadata(
            source="Google Earth Engine Sentinel-2 SR Harmonized (local raster cache)",
            year=2023,
            resolution=f"{self.selected_scale_m}m",
            p5=float(p5),
            p95=float(p95),
            path=str(self.raster_path),
        ).__dict__
