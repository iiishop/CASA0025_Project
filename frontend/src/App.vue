<script setup>
import { computed, onMounted, onBeforeUnmount, ref, watch } from "vue";
import mapboxgl from "mapbox-gl";

const backendBaseUrl = import.meta.env.VITE_BACKEND_BASE_URL || "http://127.0.0.1:8000";
const mapboxToken = import.meta.env.VITE_MAPBOX_ACCESS_TOKEN || "";

const mapContainer = ref(null);
const mapRef = ref(null);
const latestMeta = ref(null);
const errorMessage = ref("");
const loading = ref(false);
let refreshTimer = null;
let latestRequestId = 0;
let activeController = null;

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

async function fetchLatestMeta() {
  const response = await fetch(`${backendBaseUrl}/meta/air-quality/latest`);
  if (!response.ok) {
    throw new Error(`Failed to fetch metadata: ${response.status}`);
  }
  latestMeta.value = await response.json();
}

async function fetchTiles(signal) {
  const params = new URLSearchParams({
    no2_weight: String(weights.value.no2),
    pm25_weight: String(weights.value.pm25),
    pm10_weight: String(weights.value.pm10),
  });

  const response = await fetch(`${backendBaseUrl}/tiles/air-quality?${params.toString()}`, { signal });
  if (!response.ok) {
    throw new Error(`Failed to fetch tiles URL: ${response.status}`);
  }
  return response.json();
}

function initMap() {
  if (!mapboxToken) {
    throw new Error("VITE_MAPBOX_ACCESS_TOKEN is missing.");
  }

  mapboxgl.accessToken = mapboxToken;

  const map = new mapboxgl.Map({
    container: mapContainer.value,
    style: "mapbox://styles/mapbox/light-v11",
    center: [-0.1278, 51.5074],
    zoom: 9,
    pitch: 0,
    attributionControl: true,
  });

  map.addControl(new mapboxgl.NavigationControl({ visualizePitch: true }), "top-right");
  mapRef.value = map;

  map.on("load", async () => {
    await refreshAirQualityLayer();
  });
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
    const tileResponse = await fetchTiles(activeController.signal);
    if (requestId !== latestRequestId) {
      return;
    }

    const map = mapRef.value;
    const existingSource = map.getSource("air-quality-source");

    if (existingSource && typeof existingSource.setTiles === "function") {
      existingSource.setTiles([tileResponse.tile_url]);
    } else {
      map.addSource("air-quality-source", {
        type: "raster",
        tiles: [tileResponse.tile_url],
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

    if (latestMeta.value) {
      latestMeta.value.latest_image_time_utc = tileResponse.latest_image_time_utc;
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
        Live NO2 + PM2.5 + PM10 fusion from GEE CAMS. The panel is already designed as a
        reusable weighting system for future factors.
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

      <div class="card meta-card" v-if="latestMeta">
        <h2>Latest data</h2>
        <p><span>Dataset:</span> {{ latestMeta.dataset }}</p>
        <p><span>Time (UTC):</span> {{ latestMeta.latest_image_time_utc }}</p>
        <p><span>Timezone logic:</span> {{ latestMeta.timezone_reference }}</p>
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
