<script setup>
import { computed, onMounted, onBeforeUnmount, ref, watch } from "vue";
import maplibregl from "maplibre-gl";

const backendBaseUrl = import.meta.env.VITE_BACKEND_BASE_URL || "http://127.0.0.1:8000";

const mapContainer = ref(null);
const mapRef = ref(null);
const latestMeta = ref(null);
const errorMessage = ref("");
const loading = ref(false);
const osmRoadLayerEnabled = ref(false);
const osmLoading = ref(false);
const osmLoadProgress = ref(0);
const osmVisibleCount = ref(0);
const osmBytesLoaded = ref(0);
const osmBytesTotal = ref(0);
let refreshTimer = null;
let roadRefreshTimer = null;
let latestRequestId = 0;
let latestRoadRequestId = 0;
let activeController = null;
let activeRoadController = null;

const weights = ref({ no2: 0.4, pm25: 0.35, pm10: 0.25 });

const totalWeight = computed(() =>
  Number((weights.value.no2 + weights.value.pm25 + weights.value.pm10).toFixed(2)),
);

const sliderPercents = computed(() => {
  const sum = Math.max(totalWeight.value, 1e-9);
  return {
    no2: Math.round((weights.value.no2 / sum) * 100),
    pm25: Math.round((weights.value.pm25 / sum) * 100),
    pm10: Math.round((weights.value.pm10 / sum) * 100),
  };
});

async function fetchLatestMeta(signal) {
  const response = await fetch(`${backendBaseUrl}/meta/air-quality/latest`, { signal });
  if (!response.ok) {
    throw new Error(`Failed to fetch metadata: ${response.status}`);
  }
  latestMeta.value = await response.json();
}

function buildNormalTileUrl() {
  const params = new URLSearchParams({
    no2_weight: String(weights.value.no2),
    pm25_weight: String(weights.value.pm25),
    pm10_weight: String(weights.value.pm10),
  });
  return `${backendBaseUrl}/tiles/air-quality/{z}/{x}/{y}.png?${params.toString()}`;
}

function buildOsmRoadVectorUrl(map) {
  const bounds = map.getBounds();
  const zoom = Math.floor(map.getZoom());
  const params = new URLSearchParams({
    lon_min: String(bounds.getWest()),
    lat_min: String(bounds.getSouth()),
    lon_max: String(bounds.getEast()),
    lat_max: String(bounds.getNorth()),
    zoom: String(zoom),
  });
  return `${backendBaseUrl}/vectors/osm/roads?${params.toString()}`;
}

function initMap() {
  const map = new maplibregl.Map({
    container: mapContainer.value,
    style: {
      version: 8,
      sources: {
        "osm-base": {
          type: "raster",
          tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
          tileSize: 256,
          attribution: "© OpenStreetMap contributors",
          maxzoom: 19,
        },
      },
      layers: [
        {
          id: "osm-base-layer",
          type: "raster",
          source: "osm-base",
        },
      ],
    },
    center: [-0.1278, 51.5074],
    zoom: 9,
    pitch: 0,
    attributionControl: true,
  });

  map.addControl(new maplibregl.NavigationControl(), "top-right");
  mapRef.value = map;

  map.on("load", async () => {
    ensureRoadVectorLayer(map);
    await refreshAirQualityLayer();
  });

  map.on("moveend", () => {
    if (osmRoadLayerEnabled.value) {
      queueRoadRefresh();
    }
  });

  map.on("move", () => {
    if (osmRoadLayerEnabled.value) {
      queueRoadRefresh();
    }
  });

  map.on("zoomend", () => {
    if (osmRoadLayerEnabled.value) {
      queueRoadRefresh();
    }
  });
}

function ensureRoadVectorLayer(map) {
  if (!map.getSource("osm-road-vector-source")) {
    map.addSource("osm-road-vector-source", {
      type: "geojson",
      data: {
        type: "FeatureCollection",
        features: [],
      },
    });
  }

  if (!map.getLayer("osm-road-vector-layer")) {
    map.addLayer({
      id: "osm-road-vector-layer",
      type: "line",
      source: "osm-road-vector-source",
      paint: {
        "line-color": "#18d0a5",
        "line-opacity": 0.84,
        "line-width": [
          "interpolate",
          ["linear"],
          ["zoom"],
          9,
          0.7,
          12,
          1.2,
          15,
          2.4,
        ],
      },
      layout: {
        visibility: "none",
      },
    });
  }
}

function syncRoadOverlayVisibility() {
  const map = mapRef.value;
  if (!map) return;
  const visibility = osmRoadLayerEnabled.value ? "visible" : "none";
  if (map.getLayer("osm-road-vector-layer")) {
    map.setLayoutProperty("osm-road-vector-layer", "visibility", visibility);
  }
}

async function refreshRoadVectorLayer() {
  const map = mapRef.value;
  if (!map || !map.isStyleLoaded()) return;

  const requestId = ++latestRoadRequestId;
  if (activeRoadController) {
    activeRoadController.abort();
  }
  activeRoadController = new AbortController();

  osmLoading.value = true;
  osmLoadProgress.value = Math.max(osmLoadProgress.value, 3);
  osmBytesLoaded.value = 0;
  osmBytesTotal.value = 0;

  try {
    const response = await fetch(buildOsmRoadVectorUrl(map), {
      signal: activeRoadController.signal,
    });
    if (!response.ok) {
      throw new Error(`Failed to load OSM roads: ${response.status}`);
    }
    if (requestId !== latestRoadRequestId) {
      return;
    }

    const contentLength = Number(response.headers.get("content-length") || 0);
    if (Number.isFinite(contentLength) && contentLength > 0) {
      osmBytesTotal.value = contentLength;
    }

    let data;
    if (!response.body) {
      data = await response.json();
      osmLoadProgress.value = 100;
    } else {
      const reader = response.body.getReader();
      const chunks = [];
      let received = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
        osmBytesLoaded.value = received;
        if (osmBytesTotal.value > 0) {
          osmLoadProgress.value = Math.min(
            99,
            Math.round((received / osmBytesTotal.value) * 100),
          );
        } else {
          const pseudo = Math.round(100 * (1 - Math.exp(-received / 350000)));
          osmLoadProgress.value = Math.min(95, Math.max(6, pseudo));
        }
      }

      const merged = new Uint8Array(received);
      let offset = 0;
      for (const chunk of chunks) {
        merged.set(chunk, offset);
        offset += chunk.length;
      }

      const text = new TextDecoder("utf-8").decode(merged);
      data = JSON.parse(text);
      osmLoadProgress.value = 100;
      osmBytesLoaded.value = received;
      if (!osmBytesTotal.value) {
        osmBytesTotal.value = received;
      }
    }

    osmVisibleCount.value = Array.isArray(data.features) ? data.features.length : 0;
    const source = map.getSource("osm-road-vector-source");
    if (source && typeof source.setData === "function") {
      source.setData(data);
    }

    setTimeout(() => {
      if (requestId === latestRoadRequestId) {
        osmLoading.value = false;
        osmLoadProgress.value = 0;
        osmBytesLoaded.value = 0;
        osmBytesTotal.value = 0;
      }
    }, 180);
  } catch (error) {
    if (error?.name !== "AbortError") {
      errorMessage.value = error instanceof Error ? error.message : String(error);
      osmLoading.value = false;
      osmLoadProgress.value = 0;
      osmBytesLoaded.value = 0;
      osmBytesTotal.value = 0;
    }
  } finally {
    if (requestId !== latestRoadRequestId) {
      return;
    }
  }
}

function formatBytes(value) {
  if (!value || value <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let idx = 0;
  while (size >= 1024 && idx < units.length - 1) {
    size /= 1024;
    idx += 1;
  }
  return `${size.toFixed(idx === 0 ? 0 : 1)} ${units[idx]}`;
}

async function refreshAirQualityLayer() {
  if (!mapRef.value) return;

  const requestId = ++latestRequestId;
  if (activeController) {
    activeController.abort();
  }
  activeController = new AbortController();

  loading.value = true;
  errorMessage.value = "";

  try {
    await fetchLatestMeta(activeController.signal);
    if (requestId !== latestRequestId) {
      return;
    }

    const map = mapRef.value;
    const existingSource = map.getSource("air-quality-source");
    const selectedTileUrl = buildNormalTileUrl();

    if (existingSource && typeof existingSource.setTiles === "function") {
      existingSource.setTiles([selectedTileUrl]);
    } else {
      if (map.getLayer("air-quality-raster")) {
        map.removeLayer("air-quality-raster");
      }
      if (map.getSource("air-quality-source")) {
        map.removeSource("air-quality-source");
      }

      map.addSource("air-quality-source", {
        type: "raster",
        tiles: [selectedTileUrl],
        tileSize: 256,
      });

      map.addLayer({
        id: "air-quality-raster",
        type: "raster",
        source: "air-quality-source",
        paint: {
          "raster-opacity": 0.82,
          "raster-resampling": "linear",
          "raster-fade-duration": 0,
          "raster-contrast": 0,
          "raster-brightness-max": 1,
        },
      });
    }

    if (map.getLayer("air-quality-raster")) {
      map.setPaintProperty("air-quality-raster", "raster-opacity", 0.82);
    }

  } catch (error) {
    if (error?.name !== "AbortError") {
      errorMessage.value = error instanceof Error ? error.message : String(error);
    }
  } finally {
    if (requestId === latestRequestId) {
      loading.value = false;
    }
  }
}

function queueRefresh() {
  if (refreshTimer) {
    clearTimeout(refreshTimer);
  }
  refreshTimer = setTimeout(() => {
    if (mapRef.value?.isStyleLoaded()) {
      refreshAirQualityLayer();
    }
  }, 280);
}

function queueRoadRefresh() {
  osmLoading.value = true;
  osmLoadProgress.value = Math.max(osmLoadProgress.value, 2);
  if (roadRefreshTimer) {
    clearTimeout(roadRefreshTimer);
  }
  roadRefreshTimer = setTimeout(() => {
    if (mapRef.value?.isStyleLoaded()) {
      refreshRoadVectorLayer();
    }
  }, 220);
}

function toggleOsmRoadLayer() {
  osmRoadLayerEnabled.value = !osmRoadLayerEnabled.value;
  syncRoadOverlayVisibility();
  if (osmRoadLayerEnabled.value) {
    queueRoadRefresh();
  } else {
    if (activeRoadController) {
      activeRoadController.abort();
    }
    osmLoading.value = false;
    osmLoadProgress.value = 0;
    osmBytesLoaded.value = 0;
    osmBytesTotal.value = 0;
  }
}

function normalizeWeights(changedKey) {
  const current = { ...weights.value };
  const changedValue = current[changedKey];

  const otherKeys = Object.keys(current).filter((k) => k !== changedKey);
  const otherSum = otherKeys.reduce((sum, key) => sum + current[key], 0);
  const remain = Math.max(0, 1 - changedValue);

  if (otherSum <= 0) {
    const share = remain / otherKeys.length;
    otherKeys.forEach((key) => {
      current[key] = share;
    });
  } else {
    otherKeys.forEach((key) => {
      current[key] = (current[key] / otherSum) * remain;
    });
  }

  weights.value = {
    no2: Number(current.no2.toFixed(3)),
    pm25: Number(current.pm25.toFixed(3)),
    pm10: Number(current.pm10.toFixed(3)),
  };
}

function onSliderInput(key, event) {
  const value = Number(event.target.value);
  weights.value = { ...weights.value, [key]: value };
  normalizeWeights(key);
}

watch(
  () => ({ ...weights.value }),
  () => {
    queueRefresh();
  },
  { deep: true },
);

onMounted(async () => {
  try {
    await fetchLatestMeta();
    initMap();
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  }
});

onBeforeUnmount(() => {
  if (refreshTimer) {
    clearTimeout(refreshTimer);
  }
  if (roadRefreshTimer) {
    clearTimeout(roadRefreshTimer);
  }
  if (activeController) {
    activeController.abort();
  }
  if (activeRoadController) {
    activeRoadController.abort();
  }
  if (mapRef.value) {
    mapRef.value.remove();
  }
});
</script>

<template>
  <main class="page-shell">
    <section class="left-panel">
      <p class="eyebrow">ComfortPath</p>
      <h1>London Air Quality Layer</h1>
      <p class="subtitle">
        LAEI NO2 + PM2.5 + PM10 fusion is rendered as an independent air-quality layer.
        OSM roads are rendered as a separate overlay layer.
      </p>

      <div class="card">
        <h2>Weights</h2>

        <label class="slider-row">
          <span>NO2</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            :value="weights.no2"
            @input="onSliderInput('no2', $event)"
          />
          <strong>{{ sliderPercents.no2 }}%</strong>
        </label>

        <label class="slider-row">
          <span>PM2.5</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            :value="weights.pm25"
            @input="onSliderInput('pm25', $event)"
          />
          <strong>{{ sliderPercents.pm25 }}%</strong>
        </label>

        <label class="slider-row">
          <span>PM10</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            :value="weights.pm10"
            @input="onSliderInput('pm10', $event)"
          />
          <strong>{{ sliderPercents.pm10 }}%</strong>
        </label>

        <p class="sum-line">Total: {{ totalWeight }}</p>
      </div>

      <div class="card">
        <h2>OSM Roads (Vector)</h2>
        <button class="focus-button" @click="toggleOsmRoadLayer">
          {{ osmRoadLayerEnabled ? "Hide OSM Roads" : "Show OSM Roads" }}
        </button>
        <p v-if="osmRoadLayerEnabled" class="sum-line">Visible segments: {{ osmVisibleCount }}</p>
        <div v-if="osmLoading" class="progress-wrap">
          <div class="progress-track">
            <div class="progress-fill" :style="{ width: `${osmLoadProgress}%` }" />
          </div>
          <p class="progress-label">
            Refreshing OSM roads... {{ Math.round(osmLoadProgress) }}%
            ({{ formatBytes(osmBytesLoaded) }} / {{ formatBytes(osmBytesTotal) }})
          </p>
        </div>
      </div>

      <div class="card meta-card" v-if="latestMeta">
        <h2>Latest data</h2>
        <p><span>Dataset:</span> {{ latestMeta.dataset }}</p>
        <p><span>Source:</span> {{ latestMeta.source }}</p>
        <p><span>Year:</span> {{ latestMeta.year }}</p>
        <p><span>Resolution:</span> {{ latestMeta.resolution }}</p>
      </div>

      <p v-if="loading" class="status">Updating raster layer...</p>
      <p v-if="errorMessage" class="error">{{ errorMessage }}</p>
    </section>

    <section class="map-panel">
      <div ref="mapContainer" class="map" />
      <div class="legend">
        <p>Air Quality Risk</p>
        <div class="legend-gradient" />
        <div class="legend-scale">
          <span>Low</span>
          <span>High</span>
        </div>
      </div>
    </section>
  </main>
</template>
