from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    gee_project_id: str
    london_tz: str = "Europe/London"
    cams_collection: str = "ECMWF/CAMS/NRT"
    london_geojson_url: str = (
        "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
        "Regions_December_2023_Boundaries_EN_BGC/FeatureServer/0/query"
        "?where=RGN23NM%3D%27London%27"
        "&outFields=RGN23CD,RGN23NM"
        "&returnGeometry=true&outSR=4326&f=geojson"
    )


def get_settings() -> Settings:
    project_id = os.getenv("GEE_PROJECT_ID", "").strip()
    if not project_id:
        raise RuntimeError(
            "GEE_PROJECT_ID is required. Set it in environment variables."
        )

    return Settings(gee_project_id=project_id)
