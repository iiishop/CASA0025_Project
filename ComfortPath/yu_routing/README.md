# Walking Routing Prototype

This project is a walking routing prototype for London.
It combines an OpenStreetMap-based walking network, DEM-derived slope, a NetworkX graph, a Flask backend, and a Leaflet frontend.

The current prototype can:

- geocode origin and destination from typed addresses
- use browser geolocation for the origin
- allow manual map click selection
- return a Shortest Route and a Personalised Route
- demonstrate a simple user preference control through a **steepness slider**


## 1. Main workflow

The main processing workflow is:

1. extract the base walkable road network from OSM/PBF
2. add slope information from the DEM
3. filter and prepare the final routing input network
4. build and cache the main NetworkX graph
5. run the Flask + Leaflet routing prototype


## 2. Core Python files

These are the main Python files that should be kept in a clean submission.

### `export_full_network_from_pbf.py`
This script extracts the initial road network from the Greater London OSM PBF file.

What it does:

- loads the `.osm.pbf` extract with `pyrosm`
- keeps a broad walkable network
- preserves useful attributes such as `fclass`, `foot`, `sidewalk`, `service`, `u`, and `v`
- exports the canonical base road dataset

Main output:

- `data/roads_data_full_version.gpkg`


### `add_slope_to_network.py`
This script adds elevation and slope information to the base road network.

What it does:

- loads `data/roads_data_full_version.gpkg`
- samples elevation from `data/london_dem.tif`
- calculates start elevation, end elevation, elevation difference, and `slope_pct`
- classifies slope into categories such as easy / moderate / steep / unknown

Main output:

- `data/network_full_with_slope.gpkg`


### `prepare_routing_input.py`
This script prepares the slope-enriched network for routing.

What it does:

- loads `data/network_full_with_slope.gpkg`
- filters the network based on walkability rules
- keeps roads, footways, suitable paths, selected service roads, and shared cycleways
- adds `routing_reason` and `include_in_routing`
- exports the final network that is actually used for routing

Main output:

- `data/network_routing_input.gpkg`


### `build_graph_cache_from_uv.py`
This script builds the final graph used by the web app.

What it does:

- loads `data/network_routing_input.gpkg`
- creates a NetworkX graph from `u` & `v` node pairs
- stores edge attributes such as length, slope, edge type, and precomputed costs
- keeps the largest connected component only
- saves the graph as a cache file for fast loading in Flask

Main output:

- `data/main_graph.pkl`


### `routing-web/app.py`
This is the Flask backend.

What it does:

- loads the cached graph from `data/main_graph.pkl`
- handles `/geocode`, `/reverse_geocode`, and `/route`
- snaps origin and destination to the nearest graph node
- computes both the shortest route and the personalised route
- returns routes and route statistics to the frontend


### `routing-web/routing.py`
This file contains the main routing logic.

What it does:

- defines graph-building utilities
- defines edge cost logic
- builds a personalised cost function
- computes shortest paths using NetworkX


## 3. Current routing logic

### Shortest Route
The shortest route is based mainly on total distance, with a very small penalty for footpaths.

In simple terms:

`cost_shortest = length_m × type_factor`


### Personalised Route
The personalised route uses the same base distance logic but adds an uphill slope penalty.

In simple terms:

`personalised cost = length_m × type_factor × slope_penalty`

The steepness slider in the frontend changes how strong the slope penalty is.

Current slider mapping:

- `1` → lower slope sensitivity
- `2` → medium slope sensitivity
- `3` → higher slope sensitivity

Crime and noise are currently placeholders only.


## 4. Main data files in the workflow

These four files represent the main data pipeline:

- `data/roads_data_full_version.gpkg`  
  Base road network extracted from OSM/PBF

- `data/network_full_with_slope.gpkg`  
  Base road network after adding slope information from the DEM

- `data/network_routing_input.gpkg`  
  Cleaned and filtered routing-ready network

- `data/main_graph.pkl`  
  Final cached NetworkX graph used directly by the Flask app


## 5. Data source note

- **Road network**: OpenStreetMap / Geofabrik Greater London extract
- **Elevation data**: SRTM-based DEM (`data/london_dem.tif`)


## 6. Files that are now outdated or optional

These files are not part of the main submission workflow anymore:

- `build_graph_cache.py` → replaced by `build_graph_cache_from_uv.py`
- `export_full_network.py` → replaced by `export_full_network_from_pbf.py`
- `test_graph_cache.py` → test only
- `test_osmnx_london_graph.py` → test only


## 7. Running the web prototype

From the project root:

```bash
python routing-web/app.py
```

Then open:

```text
http://127.0.0.1:5000
```


## 8. GitHub submitted files

### Keep in the repo

- `export_full_network_from_pbf.py`
- `add_slope_to_network.py`
- `prepare_routing_input.py`
- `build_graph_cache_from_uv.py`
- `routing-web/app.py`
- `routing-web/routing.py`
- `routing-web/templates/index.html`
- `README.md`
- `requirements.txt`


## 9. Environment note

This project uses GIS-heavy Python packages such as `geopandas`, `rasterio`, `pyproj`, and `pyrosm`.
In practice, these are often easier to install in a Conda environment than in a plain pip-only environment.
