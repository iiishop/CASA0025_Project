from __future__ import annotations

from dataclasses import dataclass

from .laei_service import AirQualityWeights, LAEIService


@dataclass(frozen=True)
class RouteAirWeights:
    air_quality: float = 1.0
    no2: float = 0.4
    pm25: float = 0.35
    pm10: float = 0.25


class AirQualityRouteSource:
    def __init__(self, laei: LAEIService):
        self.laei = laei
        self._cache: dict[tuple[int, int, int, int], float] = {}

    def _cache_key(self, edge_id: int, w: RouteAirWeights) -> tuple[int, int, int, int]:
        return (
            edge_id,
            int(round(w.no2 * 1000)),
            int(round(w.pm25 * 1000)),
            int(round(w.pm10 * 1000)),
        )

    def edge_exposure_score(self, edge_data: dict, weights: RouteAirWeights) -> float:
        edge_id = int(edge_data["edge_id"])
        key = self._cache_key(edge_id, weights)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        score = self.laei.score_at_wgs84(
            lon=float(edge_data["mid_lon"]),
            lat=float(edge_data["mid_lat"]),
            weights=AirQualityWeights(
                no2=weights.no2,
                pm25=weights.pm25,
                pm10=weights.pm10,
            ),
        )
        self._cache[key] = score
        return score
