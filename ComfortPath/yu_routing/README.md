# CASA0025 `yu_routing` module

This folder contains Yu's routing/overlay prototype for the CASA0025 project.

It combines:

- an OpenStreetMap-derived walking network
- DEM-based slope enrichment
- a cached NetworkX graph for routing
- a Flask backend
- a Leaflet frontend with route comparison and network overlay visualisation

This README is written for the **submission version** located at:

`D:\casa0025_slope\CASA0025_Project\ComfortPath\yu_routing`

---

## 1. Folder structure

```text
yu_routing/
  data/
    main_graph.pkl
  data-prep/
    export_full_network_from_pbf.py
    add_slope_to_network.py
    prepare_routing_input.py
    merge_slope_into_anna_export.py
    export_clean_canonical_version.py
    build_graph_cache_from_uv.py
    export_graph_main_geojson.py
    verify_graph_mcdm.py
  routing-web/
    app.py
    routing.py
    templates/index.html
    static/network/
      network_tiles_manifest.json
      london_boundary.geojson
      tiles/*.geojson
  requirements.txt
  README.md
```

---

## 2. What the app does

The current web prototype can:

- geocode typed origin and destination addresses
- use browser geolocation for the origin
- compute a Shortest Route and a Personalised Route
- expose user preference sliders
- display a tiled network overlay in Leaflet

The app is designed to run locally with pre-generated runtime files.

---

## 3. Runtime files expected by the web app

The Flask app and frontend expect these local files to exist:

- `data/main_graph.pkl`
- `routing-web/static/network/network_tiles_manifest.json`
- `routing-web/static/network/tiles/*.geojson`

The overlay view may also use:

- `routing-web/static/network/london_boundary.geojson`

If these files are already present locally, the app can run without re-running the full preprocessing pipeline.

---

## 4. Run the app locally

From the `yu_routing/` root:

```bash
python routing-web/app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

## 5. Main preprocessing pipeline

The intended preprocessing order is:

1. `data-prep/export_full_network_from_pbf.py`
2. `data-prep/add_slope_to_network.py`
3. `data-prep/prepare_routing_input.py`
4. `data-prep/merge_slope_into_anna_export.py`
5. `data-prep/export_clean_canonical_version.py`
6. `data-prep/build_graph_cache_from_uv.py`
7. `data-prep/export_graph_main_geojson.py`

These scripts prepare the network, build the cached graph, and export the static overlay tiles used by the frontend.

---

## 6. Important local-data note

Some large files used by preprocessing are local-only and may not be committed to GitHub.

Examples include:

- `anna/*.gpkg`
- the Greater London OSM/PBF extract
- `data/london_dem.tif`
- generated files such as `data/main_graph.pkl`
- generated overlay assets under `routing-web/static/network/`

Therefore:

- the **web app may still run locally** if generated runtime files already exist
- the **full preprocessing workflow should not be described as one-command reproducible from a clean clone** unless the required local input files are also present

In particular, if Anna's local GPKG files are missing, several preprocessing steps cannot be re-run.

---

## 7. Key files

### `routing-web/app.py`
Flask backend entry point. Loads the cached graph and exposes `/geocode`, `/reverse_geocode`, and `/route`.

### `routing-web/routing.py`
Contains the graph/routing utilities used by the Flask app.

### `routing-web/templates/index.html`
Leaflet frontend for route display, preferences, basemap switching, and manifest-driven overlay loading.

### `data-prep/build_graph_cache_from_uv.py`
Builds the final cached NetworkX graph used by the app.

### `data-prep/export_graph_main_geojson.py`
Exports the static tiled overlay and manifest used by the frontend overlay view.

---

## 8. Environment note

This module uses GIS-heavy Python packages such as:

- `geopandas`
- `rasterio`
- `pyproj`
- `pyrosm`

In practice, these are often easier to install in a Conda-based environment than in a minimal pip-only environment.

See `requirements.txt` for the package list.

