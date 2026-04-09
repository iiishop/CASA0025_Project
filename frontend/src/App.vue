<script setup>
import { computed, onMounted, onBeforeUnmount, ref, watch } from "vue";
import mapboxgl from "mapbox-gl";

const backendBaseUrl = import.meta.env.VITE_BACKEND_BASE_URL || "http://127.0.0.1:8000";
const mapboxToken =
  import.meta.env.VITE_MAPBOX_ACCESS_TOKEN ||
  "REDACTED_MAPBOX_TOKEN";
const ROAD_BUFFER_PX = 3;

const mapContainer = ref(null);
const mapRef = ref(null);
const latestMeta = ref(null);
const errorMessage = ref("");
const loading = ref(false);
let refreshTimer = null;
let latestRequestId = 0;
let activeController = null;
let lastTileMode = "normal";

const weights = ref({ no2: 0.4, pm25: 0.35, pm10: 0.25 });
const roadFocusEnabled = ref(false);

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

function buildRoadFocusTileUrl() {
  const params = new URLSearchParams({
    no2_weight: String(weights.value.no2),
    pm25_weight: String(weights.value.pm25),
    pm10_weight: String(weights.value.pm10),
    road_buffer_px: String(ROAD_BUFFER_PX),
  });
  return `${backendBaseUrl}/tiles/air-quality-road/{z}/{x}/{y}.png?${params.toString()}`;
}

function buildOsmRoadOverlayTileUrl() {
  const params = new URLSearchParams({
    road_buffer_px: String(ROAD_BUFFER_PX),
  });
  return `${backendBaseUrl}/tiles/osm-road-overlay/{z}/{x}/{y}.png?${params.toString()}`;
}

function initMap() {
  mapboxgl.accessToken = mapboxToken;

  const map = new mapboxgl.Map({
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

  map.addControl(new mapboxgl.NavigationControl({ visualizePitch: true }), "top-right");
  mapRef.value = map;

  map.on("load", async () => {
    ensureRoadOverlayLayer(map);
    await refreshAirQualityLayer();
  });
}

function ensureRoadOverlayLayer(map) {
  if (!map.getSource("osm-road-overlay-source")) {
    map.addSource("osm-road-overlay-source", {
      type: "raster",
      tiles: [buildOsmRoadOverlayTileUrl()],
      tileSize: 256,
    });
  }

  if (!map.getLayer("osm-road-overlay-layer")) {
    map.addLayer({
      id: "osm-road-overlay-layer",
      type: "raster",
      source: "osm-road-overlay-source",
      paint: {
        "raster-opacity": 0.86,
        "raster-fade-duration": 0,
        "raster-resampling": "linear",
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
  const visibility = roadFocusEnabled.value ? "visible" : "none";
  if (map.getLayer("osm-road-overlay-layer")) {
    map.setLayoutProperty("osm-road-overlay-layer", "visibility", visibility);
  }
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
    const selectedTileUrl = roadFocusEnabled.value
      ? buildRoadFocusTileUrl()
      : buildNormalTileUrl();
    const currentMode = roadFocusEnabled.value ? "road-focus" : "normal";
    const overlayTileUrl = buildOsmRoadOverlayTileUrl();

    if (
      existingSource &&
      typeof existingSource.setTiles === "function" &&
      lastTileMode === currentMode
    ) {
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
    lastTileMode = currentMode;

    const overlaySource = map.getSource("osm-road-overlay-source");
    if (overlaySource && typeof overlaySource.setTiles === "function") {
      overlaySource.setTiles([overlayTileUrl]);
    }

    if (map.getLayer("air-quality-raster")) {
      map.setPaintProperty(
        "air-quality-raster",
        "raster-opacity",
        roadFocusEnabled.value ? 0.95 : 0.82,
      );
    }
    syncRoadOverlayVisibility();

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

function toggleRoadFocus() {
  roadFocusEnabled.value = !roadFocusEnabled.value;
  queueRefresh();
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
  if (activeController) {
    activeController.abort();
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
        LAEI NO2 + PM2.5 + PM10 fusion. Road Focus keeps the full air-quality surface,
        while smoothly increasing opacity near OSM roads and fading away from roads.
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
        <h2>Road Focus Filter</h2>
        <button class="focus-button" @click="toggleRoadFocus">
          {{ roadFocusEnabled ? "Disable Road Focus" : "Enable Road Focus" }}
        </button>
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
