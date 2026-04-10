# ComfortPath Backend (LAEI + OSM Walking)

FastAPI service serving:

- London-wide air-quality raster tiles from **LAEI 2022** concentration grids.
- OSM-based **walking route** planning with air-quality-aware weighting.

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) Data files

Expected files under `backend/data/laei/ASCII/`:

- `LAEI2022_V1_NO2.asc`
- `LAEI2022_V1_PM25.asc`
- `LAEI2022_V1_PM10m.asc`

The service reads these static 20m grids and builds weighted tiles for frontend.

Expected OSM file under `backend/data/osm/`:

- `London.osm.pbf`

On first startup, the backend builds and caches a walking graph at:

- `backend/data/osm/london_walk_graph.pkl`

OSM loading is now lazy. The graph is built on first OSM-dependent request (route/mask/meta), and progress is printed to server console via `tqdm`.

## 3) Run

```bash
uvicorn app.main:app --reload --app-dir backend --port 8000
```

Using uv from `backend/`:

```bash
uv run main.py
```

Optional env vars:

- `HOST` (default `0.0.0.0`)
- `PORT` (default `8000`)
- `RELOAD` (`true/false`, default `false`)

## 4) Endpoints

- `GET /health`
- `GET /factors`
- `GET /meta/air-quality/latest`
- `GET /tiles/air-quality`
  - query: `no2_weight`, `pm25_weight`, `pm10_weight`
  - returns tile template URL
- `GET /tiles/air-quality/{z}/{x}/{y}.png`
  - returns PNG raster tile
- `GET /meta/osm/walking`
  - returns OSM walking graph metadata
- `GET /route/walking`
  - query: `start_lon`, `start_lat`, `end_lon`, `end_lat`
  - optional query: `distance_weight`, `air_quality_weight`, `no2_weight`, `pm25_weight`, `pm10_weight`
  - returns a LineString route and summary metrics

## 5) Notes

- Data source is **LAEI 2022** (static annual concentration model, not live updates).
- Tile rendering is server-side from local ASCII rasters.
- Street routing data source is **OpenStreetMap** (`London.osm.pbf`).
- LAEI and OSM are processed in separate services and composed only at route scoring time.
