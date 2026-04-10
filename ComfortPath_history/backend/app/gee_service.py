from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import time
from typing import Any
from zoneinfo import ZoneInfo

import ee
import requests

from .config import Settings


BAND_NO2 = "total_column_nitrogen_dioxide_surface"
BAND_PM25 = "particulate_matter_d_less_than_25_um_surface"
BAND_PM10 = "particulate_matter_d_less_than_10_um_surface"

LONDONAIR_SITES_URL = "https://api.erg.ic.ac.uk/AirQuality/Information/MonitoringSiteSpecies/GroupName=London/Json"
LONDONAIR_HOURLY_INDEX_URL = (
    "https://api.erg.ic.ac.uk/AirQuality/Hourly/MonitoringIndex/GroupName=London/Json"
)
LONDONAIR_WIDE_SITE_URL = (
    "https://api.erg.ic.ac.uk/AirQuality/Data/Wide/Site/"
    "SiteCode={site}/StartDate={start}/EndDate={end}/Json"
)
LONDONAIR_DATA_URL = (
    "https://api.erg.ic.ac.uk/AirQuality/Data/SiteSpecies/"
    "SiteCode={site}/SpeciesCode={species}/StartDate={start}/EndDate={end}/Json"
)

_SENSOR_CACHE: dict[str, Any] = {"expires_at": 0.0, "payload": None}


@dataclass(frozen=True)
class AirQualityWeights:
    no2: float = 0.4
    pm25: float = 0.35
    pm10: float = 0.25


def init_ee(settings: Settings) -> None:
    ee.Initialize(project=settings.gee_project_id)


def get_london_geometry(settings: Settings) -> ee.Geometry:
    response = requests.get(settings.london_geojson_url, timeout=60)
    response.raise_for_status()
    feature_collection = response.json()
    features = feature_collection.get("features", [])
    if not features:
        raise RuntimeError("No London feature found in ONS boundary response.")

    return ee.Geometry(features[0]["geometry"])


def london_day_to_utc_window(london_tz: str) -> tuple[datetime, datetime, datetime]:
    now_local = datetime.now(ZoneInfo(london_tz))
    day_start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end_local = day_start_local + timedelta(days=1)

    start_utc = day_start_local.astimezone(timezone.utc)
    end_utc = day_end_local.astimezone(timezone.utc)
    return start_utc, end_utc, now_local


def get_latest_cams_image(settings: Settings) -> ee.Image:
    start_utc, end_utc, _ = london_day_to_utc_window(settings.london_tz)

    today_collection = ee.ImageCollection(settings.cams_collection).filterDate(
        start_utc.isoformat(), end_utc.isoformat()
    )
    fallback_collection = ee.ImageCollection(settings.cams_collection).filterDate(
        (start_utc - timedelta(days=7)).isoformat(), end_utc.isoformat()
    )

    selected = ee.ImageCollection(
        ee.Algorithms.If(
            today_collection.size().gt(0),
            today_collection,
            fallback_collection,
        )
    ).sort("system:time_start", False)

    image = ee.Image(selected.first())
    if image is None:
        raise RuntimeError("No CAMS image found for today or the previous 7 days.")

    return image


def _normalize_p10_p90(
    image: ee.Image, band_name: str, region: ee.Geometry
) -> ee.Image:
    stats = image.select([band_name]).reduceRegion(
        reducer=ee.Reducer.percentile([10, 90]),
        geometry=region,
        scale=10000,
        maxPixels=1e11,
        bestEffort=True,
    )

    p10 = ee.Number(stats.get(f"{band_name}_p10"))
    p90 = ee.Number(stats.get(f"{band_name}_p90"))
    denominator = p90.subtract(p10).max(1e-12)

    return image.select([band_name]).subtract(p10).divide(denominator).clamp(0, 1)


def _normalize_fixed(image: ee.Image, max_value: float) -> ee.Image:
    return image.divide(max_value).clamp(0, 1)


def _normalize_single_band_p10_p90(image: ee.Image, region: ee.Geometry) -> ee.Image:
    stats = image.reduceRegion(
        reducer=ee.Reducer.percentile([10, 90]),
        geometry=region,
        scale=1000,
        maxPixels=1e11,
        bestEffort=True,
    )
    p10 = ee.Number(stats.get("value_p10"))
    p90 = ee.Number(stats.get("value_p90"))
    denominator = p90.subtract(p10).max(1e-12)
    return image.subtract(p10).divide(denominator).clamp(0, 1)


def _local_date_str(dt: datetime) -> str:
    return dt.strftime("%d %b %Y")


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    except (TypeError, ValueError):
        return None
    return None


def _fetch_londonair_sites() -> list[dict[str, Any]]:
    response = requests.get(LONDONAIR_SITES_URL, timeout=60)
    response.raise_for_status()
    payload = response.json()
    sites = payload.get("Sites", {}).get("Site", [])
    if isinstance(sites, dict):
        sites = [sites]
    return sites


def _fetch_londonair_hourly_index() -> dict[str, Any]:
    response = requests.get(LONDONAIR_HOURLY_INDEX_URL, timeout=60)
    response.raise_for_status()
    return response.json()


def _fetch_londonair_site_wide(
    site_code: str, start_date: str, end_date: str
) -> dict[str, Any] | None:
    url = LONDONAIR_WIDE_SITE_URL.format(
        site=site_code,
        start=start_date.replace(" ", "%20"),
        end=end_date.replace(" ", "%20"),
    )
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        return None

    body = response.text.strip()
    if body.startswith("<"):
        return None
    return response.json()


def _extract_latest_species_from_wide(
    payload: dict[str, Any],
    now_utc: datetime,
) -> dict[str, float]:
    data_root = payload.get("AirQualityData", {})
    columns = data_root.get("Columns", {}).get("Column", [])
    if isinstance(columns, dict):
        columns = [columns]

    id_to_species: dict[str, str] = {}
    for column in columns:
        cid = column.get("@ColumnId", "")
        name = (column.get("@ColumnName", "") or "").lower()
        if "nitrogen dioxide" in name:
            id_to_species[cid] = "NO2"
        elif "pm2.5" in name:
            id_to_species[cid] = "PM25"
        elif "pm10" in name:
            id_to_species[cid] = "PM10"

    rows = data_root.get("RawAQData", {}).get("Data", [])
    if isinstance(rows, dict):
        rows = [rows]

    latest: dict[str, tuple[datetime, float]] = {}
    for row in rows:
        ts_raw = row.get("@MeasurementDateGMT")
        if not ts_raw:
            continue
        try:
            ts = datetime.strptime(ts_raw, "%Y-%m-%d %H:%M").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
        if ts > now_utc:
            continue

        for column_id, species in id_to_species.items():
            value = _safe_float(row.get(f"@{column_id}"))
            if value is None:
                continue
            prev = latest.get(species)
            if prev is None or ts > prev[0]:
                latest[species] = (ts, value)

    return {species: val for species, (_, val) in latest.items()}


def _extract_latest_measurement(
    site_code: str,
    species_code: str,
    start_date: str,
    end_date: str,
    now_utc: datetime,
) -> float | None:
    url = LONDONAIR_DATA_URL.format(
        site=site_code,
        species=species_code,
        start=start_date.replace(" ", "%20"),
        end=end_date.replace(" ", "%20"),
    )

    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        return None

    body = response.text.strip()
    if body.startswith("<"):
        return None

    records = response.json().get("RawAQData", {}).get("Data", [])
    if isinstance(records, dict):
        records = [records]

    latest_value: float | None = None
    latest_time: datetime | None = None
    for row in records:
        value = _safe_float(row.get("@Value"))
        timestamp_raw = row.get("@MeasurementDateGMT")
        if value is None or not timestamp_raw:
            continue

        try:
            timestamp = datetime.strptime(timestamp_raw, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue

        if timestamp > now_utc:
            continue

        if latest_time is None or timestamp > latest_time:
            latest_time = timestamp
            latest_value = value

    return latest_value


def _build_species_feature_collection(
    sites: list[dict[str, Any]],
    species_code: str,
    now_local: datetime,
) -> tuple[ee.FeatureCollection | None, int]:
    now_utc = now_local.astimezone(timezone.utc)
    start_date = _local_date_str(now_local - timedelta(days=1))
    end_date = _local_date_str(now_local)

    features: list[ee.Feature] = []
    for site in sites:
        site_species = site.get("Species", [])
        if isinstance(site_species, dict):
            site_species = [site_species]

        if not any(item.get("@SpeciesCode") == species_code for item in site_species):
            continue

        site_code = site.get("@SiteCode")
        lat = _safe_float(site.get("@LatitudeWGS84") or site.get("@Latitude"))
        lon = _safe_float(site.get("@LongitudeWGS84") or site.get("@Longitude"))
        if not site_code or lat is None or lon is None:
            continue

        value = _extract_latest_measurement(
            site_code=site_code,
            species_code=species_code,
            start_date=start_date,
            end_date=end_date,
            now_utc=now_utc,
        )
        if value is None:
            continue

        features.append(
            ee.Feature(
                ee.Geometry.Point([lon, lat]), {"value": value, "site": site_code}
            )
        )

    if not features:
        return None, 0
    return ee.FeatureCollection(features), len(features)


def _interpolate_sensor_image(
    feature_collection: ee.FeatureCollection,
    london: ee.Geometry,
    expected_mean: float,
    expected_std: float,
) -> ee.Image:
    del expected_mean, expected_std
    base = feature_collection.reduceToImage(["value"], ee.Reducer.mean()).rename(
        "value"
    )
    expanded = base.focal_mean(radius=2, units="pixels", iterations=28)
    return expanded.unmask(base).clip(london)


def _get_sensor_component_images(
    settings: Settings,
) -> tuple[ee.Image, ee.Image, ee.Image, dict[str, int]]:
    now_local = datetime.now(ZoneInfo(settings.london_tz))
    cache_key = now_local.strftime("%Y-%m-%d-%H")
    now_ts = time.time()

    payload = _SENSOR_CACHE.get("payload")
    if (
        _SENSOR_CACHE.get("expires_at", 0) > now_ts
        and payload
        and payload.get("key") == cache_key
    ):
        return (
            payload["no2"],
            payload["pm25"],
            payload["pm10"],
            payload["counts"],
        )

    london = get_london_geometry(settings)
    hourly = _fetch_londonair_hourly_index()

    authorities = hourly.get("HourlyAirQualityIndex", {}).get("LocalAuthority", [])
    if isinstance(authorities, dict):
        authorities = [authorities]

    species_points: dict[str, list[ee.Feature]] = {"NO2": [], "PM25": [], "PM10": []}
    now_utc = now_local.astimezone(timezone.utc)
    start_date = _local_date_str(now_local - timedelta(days=1))
    end_date = _local_date_str(now_local)

    site_wants: dict[str, dict[str, Any]] = {}
    for authority in authorities:
        sites = authority.get("Site", [])
        if isinstance(sites, dict):
            sites = [sites]

        for site in sites:
            lat = _safe_float(site.get("@LatitudeWGS84") or site.get("@Latitude"))
            lon = _safe_float(site.get("@LongitudeWGS84") or site.get("@Longitude"))
            if lat is None or lon is None:
                continue

            site_code = site.get("@SiteCode", "")
            if not site_code:
                continue
            site_species = site.get("Species", [])
            if isinstance(site_species, dict):
                site_species = [site_species]

            wants = site_wants.setdefault(
                site_code,
                {
                    "lat": lat,
                    "lon": lon,
                    "species": set(),
                },
            )

            for sp in site_species:
                code = sp.get("@SpeciesCode")
                if code not in species_points:
                    continue
                if sp.get("@AirQualityBand") == "No data":
                    continue

                wants["species"].add(code)

    for site_code, meta in site_wants.items():
        payload = _fetch_londonair_site_wide(site_code, start_date, end_date)
        if not payload:
            continue
        latest_values = _extract_latest_species_from_wide(payload, now_utc)

        lat = meta["lat"]
        lon = meta["lon"]
        for code in meta["species"]:
            value = latest_values.get(code)
            if value is None:
                continue
            species_points[code].append(
                ee.Feature(
                    ee.Geometry.Point([lon, lat]),
                    {"value": value, "site": site_code},
                )
            )

    no2_features = species_points["NO2"]
    pm25_features = species_points["PM25"]
    pm10_features = species_points["PM10"]

    no2_count = len(no2_features)
    pm25_count = len(pm25_features)
    pm10_count = len(pm10_features)

    no2_fc = ee.FeatureCollection(no2_features) if no2_features else None
    pm25_fc = ee.FeatureCollection(pm25_features) if pm25_features else None
    pm10_fc = ee.FeatureCollection(pm10_features) if pm10_features else None

    if no2_fc is None or pm25_fc is None or pm10_fc is None:
        raise RuntimeError("LondonAir returned insufficient sensor data.")

    no2_image = _interpolate_sensor_image(
        no2_fc, london, expected_mean=3, expected_std=2
    )
    pm25_image = _interpolate_sensor_image(
        pm25_fc, london, expected_mean=3, expected_std=2
    )
    pm10_image = _interpolate_sensor_image(
        pm10_fc, london, expected_mean=3, expected_std=2
    )

    payload = {
        "key": cache_key,
        "no2": no2_image,
        "pm25": pm25_image,
        "pm10": pm10_image,
        "counts": {"no2": no2_count, "pm25": pm25_count, "pm10": pm10_count},
    }
    _SENSOR_CACHE["payload"] = payload
    _SENSOR_CACHE["expires_at"] = now_ts + 900

    return no2_image, pm25_image, pm10_image, payload["counts"]


def build_air_quality_score_image(
    settings: Settings,
    weights: AirQualityWeights = AirQualityWeights(),
) -> tuple[ee.Image, ee.Image, ee.Geometry]:
    london = get_london_geometry(settings)
    source = get_latest_cams_image(settings)

    try:
        no2_sensor, pm25_sensor, pm10_sensor, _ = _get_sensor_component_images(settings)
        no2_norm = _normalize_single_band_p10_p90(no2_sensor, london)
        pm25_norm = _normalize_single_band_p10_p90(pm25_sensor, london)
        pm10_norm = _normalize_single_band_p10_p90(pm10_sensor, london)
    except Exception:
        fallback = source.select([BAND_NO2, BAND_PM25, BAND_PM10]).rename(
            ["no2", "pm25", "pm10"]
        )
        no2_norm = _normalize_p10_p90(fallback, "no2", london)
        pm25_norm = _normalize_p10_p90(fallback, "pm25", london)
        pm10_norm = _normalize_p10_p90(fallback, "pm10", london)

    weight_sum = max(weights.no2 + weights.pm25 + weights.pm10, 1e-12)
    score = (
        no2_norm.multiply(weights.no2)
        .add(pm25_norm.multiply(weights.pm25))
        .add(pm10_norm.multiply(weights.pm10))
        .divide(weight_sum)
        .multiply(100)
        .focal_mean(radius=1, units="pixels", iterations=4)
        .rename("air_quality_score")
        .clip(london)
    )

    return score, source, london


def get_latest_timestamp_utc(image: ee.Image) -> str:
    return (
        ee.Date(image.get("system:time_start")).format("YYYY-MM-dd HH:mm:ss").getInfo()
    )


def smooth_score_image(image: ee.Image) -> ee.Image:
    return image.focal_mean(radius=2, units="pixels").unmask(image)


def get_sensor_counts(settings: Settings) -> dict[str, int]:
    _, _, _, counts = _get_sensor_component_images(settings)
    return counts
