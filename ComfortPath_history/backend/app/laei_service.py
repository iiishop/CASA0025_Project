from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from rasterio.warp import transform as warp_transform


@dataclass(frozen=True)
class AirQualityWeights:
    no2: float = 0.4
    pm25: float = 0.35
    pm10: float = 0.25


class LAEIService:
    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.files = {
            "no2": data_root / "ASCII" / "LAEI2022_V1_NO2.asc",
            "pm25": data_root / "ASCII" / "LAEI2022_V1_PM25.asc",
            "pm10": data_root / "ASCII" / "LAEI2022_V1_PM10m.asc",
        }
        self.arrays: Dict[str, np.ndarray] = {}
        self.src_transform: Dict[str, object] = {}
        self.src_crs: Dict[str, object] = {}
        self.src_nodata: Dict[str, float | None] = {}
        self.percentiles: Dict[str, tuple[float, float]] = {}
        self._tile_cache: OrderedDict[tuple[int, int, int, int, int, int], bytes] = (
            OrderedDict()
        )
        self._tile_cache_limit = 512

    def _value_at_wgs84(self, key: str, lon: float, lat: float) -> float | None:
        xs, ys = warp_transform("EPSG:4326", self.src_crs[key], [lon], [lat])
        x = xs[0]
        y = ys[0]
        row, col = rowcol(self.src_transform[key], x, y)

        arr = self.arrays[key]
        if row < 0 or col < 0 or row >= arr.shape[0] or col >= arr.shape[1]:
            return None

        val = float(arr[row, col])
        nodata = self.src_nodata[key]
        if not np.isfinite(val):
            return None
        if nodata is not None and val == nodata:
            return None
        if val <= -9990:
            return None
        return val

    def load(self) -> None:
        for key, path in self.files.items():
            if not path.exists():
                raise RuntimeError(f"Missing LAEI raster: {path}")

            with rasterio.open(path) as ds:
                full = ds.read(1).astype(np.float32)
                self.arrays[key] = full
                self.src_transform[key] = ds.transform
                self.src_crs[key] = ds.crs
                self.src_nodata[key] = ds.nodata

                sample = ds.read(
                    1,
                    out_shape=(1024, 1024),
                    resampling=Resampling.bilinear,
                ).astype(np.float32)

            nodata = self.src_nodata[key]
            valid = np.isfinite(sample)
            if nodata is not None:
                valid &= sample != nodata
            valid &= sample > -9990

            if not np.any(valid):
                raise RuntimeError(f"No valid data in LAEI raster: {path}")

            values = sample[valid]
            p5, p95 = np.percentile(values, [5, 95])
            if p95 <= p5:
                p5 = float(np.min(values))
                p95 = float(np.max(values))
            self.percentiles[key] = (float(p5), float(p95))

    @staticmethod
    def tile_bounds_wgs84(z: int, x: int, y: int) -> tuple[float, float, float, float]:
        n = 2**z
        lon_left = x / n * 360.0 - 180.0
        lon_right = (x + 1) / n * 360.0 - 180.0

        def lat_from_tile(ty: int) -> float:
            import math

            val = np.pi * (1 - 2 * ty / n)
            return np.degrees(np.arctan(np.sinh(val)))

        lat_top = float(lat_from_tile(y))
        lat_bottom = float(lat_from_tile(y + 1))
        return lon_left, lat_bottom, lon_right, lat_top

    def _read_tile(self, key: str, z: int, x: int, y: int) -> np.ndarray:
        lon_min, lat_min, lon_max, lat_max = self.tile_bounds_wgs84(z, x, y)

        data = np.full((256, 256), np.nan, dtype=np.float32)
        dst_transform = from_bounds(lon_min, lat_min, lon_max, lat_max, 256, 256)

        reproject(
            source=self.arrays[key],
            destination=data,
            src_transform=self.src_transform[key],
            src_crs=self.src_crs[key],
            src_nodata=self.src_nodata[key],
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
            init_dest_nodata=True,
            num_threads=2,
        )

        nodata = self.src_nodata[key]
        if nodata is not None:
            data = np.where(data == nodata, np.nan, data)
        data = np.where(data <= -9990, np.nan, data)
        return data

    def _normalize(self, key: str, arr: np.ndarray) -> np.ndarray:
        p5, p95 = self.percentiles[key]
        denom = max(p95 - p5, 1e-6)
        return np.clip((arr - p5) / denom, 0.0, 1.0)

    @staticmethod
    def _colorize(score_0_1: np.ndarray, valid: np.ndarray) -> np.ndarray:
        score_safe = np.nan_to_num(score_0_1, nan=0.0, posinf=1.0, neginf=0.0)
        stops = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        colors = np.array(
            [
                [44, 123, 182],
                [167, 215, 240],
                [254, 243, 199],
                [249, 115, 22],
                [185, 28, 28],
            ],
            dtype=np.float32,
        )

        r = np.interp(score_safe, stops, colors[:, 0])
        g = np.interp(score_safe, stops, colors[:, 1])
        b = np.interp(score_safe, stops, colors[:, 2])
        a = np.where(valid, 220, 0)

        rgba = np.stack([r, g, b, a], axis=-1).astype(np.uint8)
        return rgba

    def render_tile_png(
        self, z: int, x: int, y: int, weights: AirQualityWeights
    ) -> bytes:
        cache_key = (
            z,
            x,
            y,
            int(round(weights.no2 * 1000)),
            int(round(weights.pm25 * 1000)),
            int(round(weights.pm10 * 1000)),
        )
        cached = self._tile_cache.get(cache_key)
        if cached is not None:
            self._tile_cache.move_to_end(cache_key)
            return cached

        no2 = self._read_tile("no2", z, x, y)
        pm25 = self._read_tile("pm25", z, x, y)
        pm10 = self._read_tile("pm10", z, x, y)

        no2_n = self._normalize("no2", no2)
        pm25_n = self._normalize("pm25", pm25)
        pm10_n = self._normalize("pm10", pm10)

        w_sum = max(weights.no2 + weights.pm25 + weights.pm10, 1e-9)
        score = (
            no2_n * weights.no2 + pm25_n * weights.pm25 + pm10_n * weights.pm10
        ) / w_sum

        valid = np.isfinite(no2) | np.isfinite(pm25) | np.isfinite(pm10)
        rgba = self._colorize(score, valid)
        image = Image.fromarray(rgba, mode="RGBA")
        buf = BytesIO()
        image.save(buf, format="PNG")
        value = buf.getvalue()
        self._tile_cache[cache_key] = value
        if len(self._tile_cache) > self._tile_cache_limit:
            self._tile_cache.popitem(last=False)
        return value

    def metadata(self) -> dict:
        return {
            "dataset": "LAEI 2022 Concentrations",
            "source": "GLA London Datastore",
            "year": 2022,
            "resolution": "20m grid",
            "note": "Static modeled annual mean concentrations",
            "bands": {
                "no2": "LAEI2022_V1_NO2.asc",
                "pm25": "LAEI2022_V1_PM25.asc",
                "pm10": "LAEI2022_V1_PM10m.asc",
            },
            "normalization_percentiles": {
                "no2": self.percentiles.get("no2"),
                "pm25": self.percentiles.get("pm25"),
                "pm10": self.percentiles.get("pm10"),
            },
        }

    def score_at_wgs84(
        self, lon: float, lat: float, weights: AirQualityWeights
    ) -> float:
        no2 = self._value_at_wgs84("no2", lon, lat)
        pm25 = self._value_at_wgs84("pm25", lon, lat)
        pm10 = self._value_at_wgs84("pm10", lon, lat)

        values: list[tuple[str, float, float]] = [
            ("no2", weights.no2, no2),
            ("pm25", weights.pm25, pm25),
            ("pm10", weights.pm10, pm10),
        ]

        weighted_sum = 0.0
        used_weight = 0.0
        for key, w, v in values:
            if v is None:
                continue
            v_n = float(self._normalize(key, np.array([v], dtype=np.float32))[0])
            weighted_sum += v_n * w
            used_weight += w

        if used_weight <= 0.0:
            return 0.0
        return weighted_sum / used_weight
