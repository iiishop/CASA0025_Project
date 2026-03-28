# ComfortPath Backend (LAEI)

FastAPI service serving London-wide air-quality raster tiles from **LAEI 2022** concentration grids.

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

## 5) Notes

- Data source is **LAEI 2022** (static annual concentration model, not live updates).
- Tile rendering is server-side from local ASCII rasters.
