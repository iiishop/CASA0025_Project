# Frontend (Vue + Mapbox)

This folder is intentionally separated from `backend/`.

## Stack

- Vue 3 + Vite
- `mapbox-gl` for base map and raster rendering

## Setup

1. Install dependencies:

```bash
cd frontend
npm install
```

2. Copy env file and fill your token:

```bash
cp .env.example .env
```

3. Start dev server:

```bash
npm run dev
```

## Backend integration

The app calls:
- `GET /meta/air-quality/latest`
- `GET /tiles/air-quality?no2_weight=...&pm25_weight=...&pm10_weight=...`

It then renders backend `tile_url` as a Mapbox raster layer.
