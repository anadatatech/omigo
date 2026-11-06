import asyncio
import json
import math
import struct
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import rasterio
import requests
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from pyproj import Transformer
from shapely.geometry import Point
from shapely.ops import nearest_points

app = FastAPI(title="OmiGo Core Engine — Full Lagoon GeoTIFF + OSM Base")

DATA_DIR = Path("data")
GEOJSON_DIR = DATA_DIR / "geojson"
BATHY_PATH = DATA_DIR / "bathymetry_lagos_poc.tif"

WGS84 = "EPSG:4326"
UTM31N = "EPSG:32631"

to_utm = Transformer.from_crs(WGS84, UTM31N, always_xy=True)
to_wgs = Transformer.from_crs(UTM31N, WGS84, always_xy=True)

DRAUGHT_METERS = {
    "shallow_0.8m": 0.8,
    "medium_1.5m": 1.5,
    "deep_2.5m": 2.5,
}

# =====================================================================
# PILLAR 3: OSM LAGOON WATERWAY BASE & WATER-LOCKED TRAJECTORY
# =====================================================================
water_gdf = gpd.read_file(GEOJSON_DIR / "lagos_waterways.geojson").to_crs(UTM31N)
WATER_UNION_UTM = water_gdf.geometry.union_all()
NAVIGABLE_WATER_UTM = WATER_UNION_UTM.buffer(-20.0)

pilot_gdf_wgs = gpd.read_file(GEOJSON_DIR / "pilot_corridor_route.geojson")
pilot_gdf_utm = pilot_gdf_wgs.to_crs(UTM31N)
ROUTE_LINE_UTM = pilot_gdf_utm.geometry.iloc[0]
ROUTE_LENGTH_M = ROUTE_LINE_UTM.length

# =====================================================================
# PILLAR 2: FULL-LAGOON BATHYMETRY GEOTIFF ENGINE & PNG RENDERER
# =====================================================================
BATHY_DS = rasterio.open(BATHY_PATH)
BATHY_ARRAY = BATHY_DS.read(1)
BATHY_BOUNDS = BATHY_DS.bounds  # (west, south, east, north) in EPSG:4326
BATHY_NODATA = BATHY_DS.nodata


def encode_rgba_png(rgba: np.ndarray) -> bytes:
    """Fast stdlib PNG encoder for (H, W, 4) uint8 numpy arrays."""
    h, w, _ = rgba.shape
    raw = np.hstack([np.zeros((h, 1), dtype=np.uint8), rgba.reshape(h, w * 4)]).tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack("!IIBBBBB", w, h, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


@app.get("/api/bathymetry/meta")
def bathymetry_meta():
    return {
        "bounds": [[BATHY_BOUNDS.bottom, BATHY_BOUNDS.left], [BATHY_BOUNDS.top, BATHY_BOUNDS.right]],
        "width": BATHY_DS.width,
        "height": BATHY_DS.height,
        "crs": str(BATHY_DS.crs),
    }


@app.get("/api/bathymetry.png")
def render_bathymetry_png(
    mode: Literal["depth", "operability"] = Query("depth"),
    draught_class: Literal["shallow_0.8m", "medium_1.5m", "deep_2.5m"] = Query("medium_1.5m"),
):
    """Renders the Bathymetry GeoTIFF as a transparent RGBA overlay over the OSM Lagoon."""
    d = BATHY_ARRAY
    valid = (d != BATHY_NODATA) & (~np.isnan(d)) & (d > 0)
    h, w = d.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    if mode == "depth":
        # Continuous hydrographic depth color ramp:
        # < 1.0m: Red/Orange | 1.0 - 1.8m: Amber/Yellow | 1.8 - 2.6m: Light Cyan | 2.6 - 3.4m: Mid Blue | > 3.4m: Deep Navy Blue
        b0 = valid & (d < 1.0)
        b1 = valid & (d >= 1.0) & (d < 1.8)
        b2 = valid & (d >= 1.8) & (d < 2.6)
        b3 = valid & (d >= 2.6) & (d < 3.4)
        b4 = valid & (d >= 3.4)

        rgba[b0] = [239, 68, 68, 195]    # Shallow shoal / bank (<1.0m)
        rgba[b1] = [245, 158, 11, 185]   # Caution shallows (1.0 - 1.8m)
        rgba[b2] = [56, 189, 248, 180]   # Mid-depth lagoon (1.8 - 2.6m)
        rgba[b3] = [2, 132, 199, 190]    # Navigable channel (2.6 - 3.4m)
        rgba[b4] = [30, 58, 138, 210]    # Deep channel (>3.4m)
    else:
        # Operability mask for the selected vessel draught (including squat + margin)
        draught_m = DRAUGHT_METERS[draught_class]
        wx = get_corridor_weather()
        wave_m = float(wx["current"].get("est_wave_m", 0.1))
        req_m = draught_m + 0.20 + 0.5 * wave_m + 0.40

        unsafe = valid & (d < req_m)
        caution = valid & (d >= req_m) & (d < req_m + 0.40)
        safe = valid & (d >= req_m + 0.40)

        rgba[unsafe] = [220, 38, 38, 200]   # Red: Too shallow for this boat
        rgba[caution] = [234, 179, 8, 190]  # Amber: Marginal clearance (<0.4m UKC)
        rgba[safe] = [22, 163, 74, 185]     # Green: Operationally safe depth

    return Response(content=encode_rgba_png(rgba), media_type="image/png")


def sample_depth_wgs84(lon: float, lat: float) -> float | None:
    if not (BATHY_BOUNDS.left <= lon <= BATHY_BOUNDS.right and BATHY_BOUNDS.bottom <= lat <= BATHY_BOUNDS.top):
        return None
    val = float(next(BATHY_DS.sample([(lon, lat)]))[0])
    if val == BATHY_NODATA or math.isnan(val) or val <= 0:
        return None
    return round(val, 2)


def evaluate_bathymetry(lon: float, lat: float, x_utm: float, y_utm: float, heading_deg: float, draught_m: float, wave_m: float) -> dict:
    depth_here = sample_depth_wgs84(lon, lat)

    rad = math.radians(heading_deg)
    ahead_x = x_utm + 200.0 * math.sin(rad)
    ahead_y = y_utm + 200.0 * math.cos(rad)
    ahead_lon, ahead_lat = to_wgs.transform(ahead_x, ahead_y)
    depth_ahead_200m = sample_depth_wgs84(ahead_lon, ahead_lat)

    required_depth_m = round(draught_m + 0.20 + 0.5 * wave_m + 0.40, 2)
    ukc_here = round(depth_here - required_depth_m, 2) if depth_here is not None else None
    ukc_ahead = round(depth_ahead_200m - required_depth_m, 2) if depth_ahead_200m is not None else None

    if depth_here is None:
        bathy_status = "OUTSIDE_WATERWAY"
    elif ukc_here < 0:
        bathy_status = "UNSAFE_SHALLOW_WATER"
    elif ukc_ahead is not None and ukc_ahead < 0:
        bathy_status = "SHOAL_AHEAD_200M_WARNING"
    elif ukc_here < 0.4:
        bathy_status = "MARGINAL_DEPTH_CAUTION"
    else:
        bathy_status = "OPERATIONALLY_SAFE_DEPTH"

    return {
        "depth_under_keel_m": depth_here,
        "depth_ahead_200m_m": depth_ahead_200m,
        "required_depth_m": required_depth_m,
        "ukc_m": ukc_here,
        "bathy_status": bathy_status,
    }


# =====================================================================
# PILLAR 4: OPEN-METEO WEATHER & MARINE ENGINE (10-MIN CACHE)
# =====================================================================
FETCH_DISTANCE_M = 5000.0  # Placeholder fetch distance
WEATHER_POINTS = [
    {"id": "ikorodu", "name": "Ikorodu", "lat": 6.60086, "lon": 3.48818},
    {"id": "banana_island", "name": "Mid-Lagoon", "lat": 6.46700, "lon": 3.45000},
    {"id": "falomo", "name": "Falomo", "lat": 6.44248, "lon": 3.42725},
]
WEATHER_CACHE: dict = {"fetched_at_epoch": 0.0, "data": None}


def estimate_wave_m(wind_kn: float) -> float:
    return round(0.0016 * (wind_kn * 0.5144) * math.sqrt(FETCH_DISTANCE_M / 9.81), 2)


def assess_single_hour(wind_kn: float, gust_kn: float, wave_m: float, vis_m: float, cape: float, rain_prob: float, code: int, hour_local: int):
    order = {"GO": 0, "HOLD": 1, "STOP": 2}
    level, reasons = "GO", []

    def escalate(new_lvl: str, reason: str):
        nonlocal level
        if order[new_lvl] > order[level]:
            level = new_lvl
        if reason not in reasons:
            reasons.append(reason)

    if wind_kn >= 22: escalate("STOP", f"Wind {wind_kn} kn")
    elif wind_kn >= 15: escalate("HOLD", f"Wind {wind_kn} kn")
    if gust_kn >= 30: escalate("STOP", f"Gusts {gust_kn} kn")
    elif gust_kn >= 22: escalate("HOLD", f"Gusts {gust_kn} kn")
    if wave_m >= 0.8: escalate("STOP", f"Waves {wave_m} m")
    elif wave_m >= 0.5: escalate("HOLD", f"Waves {wave_m} m")
    if vis_m <= 1000: escalate("STOP", f"Vis {int(vis_m)} m")
    elif vis_m <= 2000: escalate("HOLD", f"Vis {int(vis_m)} m")
    if code in (95, 96, 97, 98, 99): escalate("STOP", f"Thunderstorm (WMO {code})")
    elif (cape >= 1500 and rain_prob >= 50) or code in (65, 82): escalate("HOLD", f"Storm risk (CAPE {int(cape)})")
    if not (6 <= hour_local < 19): escalate("STOP", "Night hours")
    return level, reasons


def get_corridor_weather() -> dict:
    now_epoch = time.time()
    if WEATHER_CACHE["data"] is not None and (now_epoch - WEATHER_CACHE["fetched_at_epoch"]) < 600:
        cached = WEATHER_CACHE["data"].copy()
        cached["cache_age_sec"] = int(now_epoch - WEATHER_CACHE["fetched_at_epoch"])
        return cached
    try:
        f_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": ",".join(str(p["lat"]) for p in WEATHER_POINTS),
                "longitude": ",".join(str(p["lon"]) for p in WEATHER_POINTS),
                "hourly": "wind_speed_10m,wind_gusts_10m,wind_direction_10m,visibility,weather_code,precipitation_probability,precipitation,cape",
                "wind_speed_unit": "kn",
                "timezone": "Africa/Lagos",
                "forecast_days": 2,
            },
            timeout=10,
        )
        f_resp.raise_for_status()
        f_json = f_resp.json()

        m_resp = requests.get(
            "https://marine-api.open-meteo.com/v1/marine",
            params={"latitude": 6.44248, "longitude": 3.42725, "hourly": "sea_level_height_msl", "cell_selection": "sea", "timezone": "Africa/Lagos", "forecast_days": 2},
            timeout=10,
        )
        m_resp.raise_for_status()
        m_json = m_resp.json()

        times = f_json[0]["hourly"]["time"]
        sea_levels = m_json.get("hourly", {}).get("sea_level_height_msl", [None] * len(times))
        now_lagos_prefix = datetime.now().strftime("%Y-%m-%dT%H:00")
        now_idx = times.index(now_lagos_prefix) if now_lagos_prefix in times else 0

        order = {"GO": 0, "HOLD": 1, "STOP": 2}
        timeline = []
        for i, t_str in enumerate(times):
            hour_local = int(t_str.split("T")[1].split(":")[0])
            worst_level, all_reasons = "GO", []
            max_wind, max_gust, max_wave, min_vis = 0.0, 0.0, 0.0, 99999.0
            for p_idx in range(len(WEATHER_POINTS)):
                h = f_json[p_idx]["hourly"]
                w_kn = float(h["wind_speed_10m"][i] or 0.0)
                g_kn = float(h["wind_gusts_10m"][i] or 0.0)
                vis_m = float(h["visibility"][i] or 10000.0)
                code = int(h["weather_code"][i] or 0)
                r_prob = float(h["precipitation_probability"][i] or 0.0)
                cape = float(h["cape"][i] or 0.0)
                wave_m = estimate_wave_m(w_kn)
                lvl, r_list = assess_single_hour(w_kn, g_kn, wave_m, vis_m, cape, r_prob, code, hour_local)
                if order[lvl] > order[worst_level]: worst_level = lvl
                for r in r_list:
                    if r not in all_reasons: all_reasons.append(r)
                max_wind, max_gust, max_wave, min_vis = max(max_wind, w_kn), max(max_gust, g_kn), max(max_wave, wave_m), min(min_vis, vis_m)

            timeline.append({
                "time": t_str,
                "signal": worst_level,
                "reasons": all_reasons or ["Conditions within threshold"],
                "wind_kn": round(max_wind, 1),
                "gust_kn": round(max_gust, 1),
                "est_wave_m": round(max_wave, 2),
                "visibility_km": round(min_vis / 1000.0, 1),
                "modelled_sea_level_msl_m": sea_levels[i] if i < len(sea_levels) else None,
            })

        payload = {"status": "LIVE", "cache_age_sec": 0, "current": timeline[now_idx], "next_12h": timeline[now_idx : now_idx + 12]}
        WEATHER_CACHE["fetched_at_epoch"] = now_epoch
        WEATHER_CACHE["data"] = payload
        return payload
    except Exception as exc:
        if WEATHER_CACHE["data"] is not None:
            return WEATHER_CACHE["data"]
        return {"status": f"OFFLINE ({exc})", "current": {"signal": "HOLD", "reasons": ["Weather offline"], "wind_kn": 0, "gust_kn": 0, "est_wave_m": 0.1, "visibility_km": 10}, "next_12h": []}


# =====================================================================
# PILLAR 1: WATERWAY-WIDE VESSEL TRACKING ENGINE
# =====================================================================
class VesselPing(BaseModel):
    vessel_id: str
    name: str
    draught_class: Literal["shallow_0.8m", "medium_1.5m", "deep_2.5m"] = "medium_1.5m"
    lat: float
    lon: float
    speed_kn: float = 14.0
    heading_deg: float = 195.0
    source: Literal["simulated", "live_gps", "manual_test"] = "live_gps"


VESSELS: dict[str, dict] = {}


def evaluate_vessel_position(ping: VesselPing) -> dict:
    x, y = to_utm.transform(ping.lon, ping.lat)
    pt_utm = Point(x, y)

    on_water = bool(WATER_UNION_UTM.contains(pt_utm))
    along_m = ROUTE_LINE_UTM.project(pt_utm)
    remaining_nm = max(0.0, (ROUTE_LENGTH_M - along_m) / 1852.0)
    eta_min = round((remaining_nm / ping.speed_kn) * 60.0, 1) if ping.speed_kn > 0.5 else None

    wx = get_corridor_weather()
    wave_m = float(wx["current"].get("est_wave_m", 0.1))
    draught_m = DRAUGHT_METERS[ping.draught_class]
    bathy = evaluate_bathymetry(ping.lon, ping.lat, x, y, ping.heading_deg, draught_m, wave_m)

    # Status is driven by actual water & bathymetric depth at point P
    if not on_water or bathy["bathy_status"] in ("OUTSIDE_WATERWAY", "UNSAFE_SHALLOW_WATER"):
        status = "UNSAFE_SHALLOW_OR_GROUNDING"
    elif bathy["bathy_status"] in ("SHOAL_AHEAD_200M_WARNING", "MARGINAL_DEPTH_CAUTION"):
        status = bathy["bathy_status"]
    else:
        status = "OPERATIONALLY_SAFE_DEPTH"

    return {
        "vessel_id": ping.vessel_id,
        "name": ping.name,
        "draught_class": ping.draught_class,
        "draught_m": draught_m,
        "lat": round(ping.lat, 6),
        "lon": round(ping.lon, 6),
        "speed_kn": round(ping.speed_kn, 1),
        "heading_deg": round(ping.heading_deg, 0),
        "progress_pct": round((along_m / ROUTE_LENGTH_M) * 100.0, 1),
        "remaining_nm": round(remaining_nm, 2),
        "eta_falomo_min": eta_min,
        "on_mapped_water": on_water,
        "status": status,
        **bathy,
    }


@app.post("/api/vessels/ping")
def ingest_ping(ping: VesselPing):
    state = evaluate_vessel_position(ping)
    VESSELS[ping.vessel_id] = state
    return state


@app.get("/api/vessels")
def list_vessels():
    return {"vessels": list(VESSELS.values())}


@app.get("/api/weather")
def weather_endpoint():
    return get_corridor_weather()


@app.get("/api/layers")
def get_layers():
    return {
        "waterways": json.loads((GEOJSON_DIR / "lagos_waterways.geojson").read_text()),
        "osm_routes": json.loads((GEOJSON_DIR / "lagos_ferry_routes.geojson").read_text()) if (GEOJSON_DIR / "lagos_ferry_routes.geojson").exists() else None,
        "terminals": json.loads((GEOJSON_DIR / "lagos_ferry_terminals.geojson").read_text()) if (GEOJSON_DIR / "lagos_ferry_terminals.geojson").exists() else None,
    }


async def simulate_fleet():
    sim_boats = [
        {"id": "LAG-01", "name": "Eko Express 1 (Deep 2.5m)", "draught": "deep_2.5m", "pos_m": 1200.0, "speed_kn": 16.0, "wander_m": 0.0},
        {"id": "LAG-02", "name": "Ikorodu Queen (Medium 1.5m)", "draught": "medium_1.5m", "pos_m": 7500.0, "speed_kn": 14.0, "wander_m": 180.0},
        {"id": "LAG-03", "name": "Cowrie Water Taxi (Shallow 0.8m)", "draught": "shallow_0.8m", "pos_m": 13500.0, "speed_kn": 18.0, "wander_m": 90.0},
    ]
    step = 0
    while True:
        step += 1
        for b in sim_boats:
            prev_pt = ROUTE_LINE_UTM.interpolate(b["pos_m"])
            b["pos_m"] = (b["pos_m"] + b["speed_kn"] * 0.5144 * 8.0) % ROUTE_LENGTH_M
            pt_on_line = ROUTE_LINE_UTM.interpolate(b["pos_m"])

            dx, dy = pt_on_line.x - prev_pt.x, pt_on_line.y - prev_pt.y
            heading_deg = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0

            # Lateral wander in water, strictly clamped inside NAVIGABLE_WATER_UTM so boats never touch land
            offset_m = math.sin(step / 6.0) * b["wander_m"]
            cand_pt = Point(pt_on_line.x + offset_m, pt_on_line.y)
            if not NAVIGABLE_WATER_UTM.contains(cand_pt):
                cand_pt = nearest_points(NAVIGABLE_WATER_UTM, cand_pt)[0]

            lon, lat = to_wgs.transform(cand_pt.x, cand_pt.y)
            ping = VesselPing(vessel_id=b["id"], name=b["name"], draught_class=b["draught"], lat=lat, lon=lon, speed_kn=b["speed_kn"], heading_deg=heading_deg, source="simulated")
            VESSELS[b["id"]] = evaluate_vessel_position(ping)
        await asyncio.sleep(2.0)


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(simulate_fleet())


@app.get("/", response_class=HTMLResponse)
def interactive_map():
    return """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>OmiGo — Full-Lagoon Bathymetry GeoTIFF & Waterway Tracker</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: system-ui, sans-serif; display: flex; height: 100vh; width: 100vw; overflow: hidden; background: #0f172a; color: #f8fafc; }
    #sidebar { width: 420px; height: 100vh; padding: 14px; overflow-y: auto; border-right: 1px solid #334155; flex-shrink: 0; }
    #map { flex: 1; height: 100vh; width: 100%; background: #1e293b; }
    h2 { margin: 0 0 2px 0; font-size: 17px; }
    .sub { font-size: 11px; color: #94a3b8; margin-bottom: 10px; }
    .card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 10px; margin-bottom: 10px; font-size: 12.5px; }
    .badge { display: inline-block; padding: 2px 7px; border-radius: 4px; font-weight: 700; font-size: 10.5px; }
    .OPERATIONALLY_SAFE_DEPTH, .GO { background: #166534; color: #dcfce7; }
    .MARGINAL_DEPTH_CAUTION, .SHOAL_AHEAD_200M_WARNING, .HOLD { background: #854d0e; color: #fef9c3; }
    .UNSAFE_SHALLOW_OR_GROUNDING, .STOP { background: #991b1b; color: #fee2e2; }
    select { width: 100%; padding: 6px; margin-top: 4px; background: #0f172a; color: white; border: 1px solid #475569; border-radius: 4px; }
    .legend { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; font-size: 11px; }
    .swatch { display: inline-block; width: 12px; height: 12px; border-radius: 2px; vertical-align: middle; margin-right: 4px; }
  </style>
</head>
<body>
  <div id="sidebar">
    <h2>OmiGo Lagoon Bathymetry & Tracker</h2>
    <div class="sub">OSM Waterway Base + Full-Lagoon GeoTIFF Depth Raster (POC)</div>

    <div class="card">
      <strong>1. Full-Lagoon Bathymetry GeoTIFF Display</strong>
      <select id="bathyMode" onchange="updateBathyOverlay()">
        <option value="depth" selected>Hydrographic Depth Map (m below Datum)</option>
        <option value="operability">Vessel Go / No-Go Depth Mask (Safe vs Shallow)</option>
      </select>
      <div id="legendBox" class="legend"></div>
    </div>

    <div class="card">
      <strong>2. Select Vessel Draught & Click Any Point P on Water</strong>
      <select id="draughtSelect" onchange="updateBathyOverlay()">
        <option value="shallow_0.8m">Shallow Boat (0.8 m draught)</option>
        <option value="medium_1.5m" selected>Medium Ferry (1.5 m draught)</option>
        <option value="deep_2.5m">Deep Vessel (2.5 m draught)</option>
      </select>
    </div>

    <div id="weatherBox" class="card">Loading live Open-Meteo weather...</div>
    <div id="vesselList">Loading vessels...</div>
  </div>
  <div id="map"></div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const osmStandard = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '&copy; OpenStreetMap' });
    const esriSatellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', { maxZoom: 19, attribution: 'Tiles &copy; Esri' });
    const map = L.map('map', { center: [6.51, 3.455], zoom: 12, layers: [osmStandard] });
    const layerControl = L.control.layers({ "OpenStreetMap": osmStandard, "Satellite (Esri)": esriSatellite }, {}, { collapsed: false }).addTo(map);

    let bathyOverlay = null;
    let bathyBounds = null;
    const markers = {};

    function renderLegend(mode) {
      const el = document.getElementById('legendBox');
      if (mode === 'depth') {
        el.innerHTML = `
          <span><i class="swatch" style="background:#ef4444"></i>&lt;1.0m Shoal</span>
          <span><i class="swatch" style="background:#f59e0b"></i>1.0–1.8m Shallow</span>
          <span><i class="swatch" style="background:#38bdf8"></i>1.8–2.6m Mid</span>
          <span><i class="swatch" style="background:#0284c7"></i>2.6–3.4m Channel</span>
          <span><i class="swatch" style="background:#1e3a8a"></i>&gt;3.4m Deep</span>
        `;
      } else {
        el.innerHTML = `
          <span><i class="swatch" style="background:#16a34a"></i>Operationally Safe Depth</span>
          <span><i class="swatch" style="background:#eab308"></i>Marginal Clearance (&lt;0.4m)</span>
          <span><i class="swatch" style="background:#dc2626"></i>Unsafe / Grounding Risk</span>
        `;
      }
    }

    function updateBathyOverlay() {
      if (!bathyBounds) return;
      const mode = document.getElementById('bathyMode').value;
      const draught = document.getElementById('draughtSelect').value;
      renderLegend(mode);
      const url = `/api/bathymetry.png?mode=${mode}&draught_class=${draught}&t=${Date.now()}`;
      if (!bathyOverlay) {
        bathyOverlay = L.imageOverlay(url, bathyBounds, { opacity: 0.78, interactive: false }).addTo(map);
        layerControl.addOverlay(bathyOverlay, "Bathymetry GeoTIFF Raster");
      } else {
        bathyOverlay.setUrl(url);
      }
    }

    fetch('/api/bathymetry/meta').then(r => r.json()).then(meta => {
      bathyBounds = meta.bounds;
      updateBathyOverlay();
    });

    fetch('/api/layers').then(r => r.json()).then(layers => {
      if (layers.waterways) {
        const wl = L.geoJSON(layers.waterways, { style: { color: '#0369a1', weight: 1.5, fillOpacity: 0.0 } }).addTo(map);
        layerControl.addOverlay(wl, "OSM Lagoon Shoreline Boundary");
      }
      if (layers.osm_routes) {
        const rl = L.geoJSON(layers.osm_routes, { style: { color: '#f8fafc', weight: 1.5, dashArray: '4,6', opacity: 0.85 } }).addTo(map);
        layerControl.addOverlay(rl, "OSM Ferry Trajectories");
      }
      if (layers.terminals) {
        const tl = L.geoJSON(layers.terminals, {
          pointToLayer: (f, latlng) => L.circleMarker(latlng, { radius: 4, color: '#0f172a', weight: 1, fillColor: '#f97316', fillOpacity: 1 }),
          onEachFeature: (f, l) => l.bindTooltip("Jetty: " + (f.properties?.name || 'Terminal'))
        }).addTo(map);
        layerControl.addOverlay(tl, "80 Ferry Terminals");
      }
    });

    map.on('click', e => {
      const draught = document.getElementById('draughtSelect').value;
      fetch('/api/vessels/ping', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          vessel_id: 'POINT-P',
          name: 'Inspected Point P',
          draught_class: draught,
          lat: e.latlng.lat,
          lon: e.latlng.lng,
          speed_kn: 14.0,
          heading_deg: 195.0,
          source: 'manual_test'
        })
      }).then(refreshVessels);
    });

    function refreshWeather() {
      fetch('/api/weather').then(r => r.json()).then(wx => {
        const cur = wx.current;
        const stripHtml = (wx.next_12h || []).map(h => `
          <div style="min-width:46px;text-align:center;padding:4px;border-radius:4px;font-size:10px;background:#0f172a;border:1px solid #334155;">
            <div>${h.time.split('T')[1]}</div>
            <div class="badge ${h.signal}" style="margin:2px 0;">${h.signal}</div>
            <div>${h.wind_kn}kn</div>
          </div>
        `).join('');

        document.getElementById('weatherBox').innerHTML = `
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <strong>3. Live Corridor Weather & Marine</strong>
            <span class="badge ${cur.signal}" style="font-size:12px;">${cur.signal}</span>
          </div>
          <div style="margin-top:6px;color:#cbd5e1;line-height:1.45;">
            Trigger: <strong>${cur.reasons.join(', ')}</strong><br/>
            Wind: <strong>${cur.wind_kn} kn</strong> · Gusts: <strong>${cur.gust_kn} kn</strong> · Est. Waves: <strong>${cur.est_wave_m} m</strong><br/>
            Visibility: <strong>${cur.visibility_km} km</strong> · Sea Level (MSL): <strong>${cur.modelled_sea_level_msl_m ?? '—'} m</strong>
          </div>
          <div style="display:flex;gap:4px;overflow-x:auto;margin-top:8px;padding-bottom:4px;">
            ${stripHtml}
          </div>
        `;
      });
    }

    function statusColor(status) {
      if (status === 'OPERATIONALLY_SAFE_DEPTH') return '#22c55e';
      if (status === 'MARGINAL_DEPTH_CAUTION' || status === 'SHOAL_AHEAD_200M_WARNING') return '#eab308';
      return '#ef4444';
    }

    function refreshVessels() {
      fetch('/api/vessels').then(r => r.json()).then(data => {
        document.getElementById('vesselList').innerHTML = data.vessels.map(v => `
          <div class="card">
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <strong>${v.name}</strong>
              <span class="badge ${v.status}">${v.status}</span>
            </div>
            <div style="margin-top:6px;color:#cbd5e1;">
              Point P Depth: <strong>${v.depth_under_keel_m ?? 'Land / Outside Water'} m</strong> (200m Ahead: <strong>${v.depth_ahead_200m_m ?? '—'} m</strong>)<br/>
              Boat Draught: ${v.draught_m} m · Required Depth: <strong>${v.required_depth_m} m</strong> · UKC: <strong>${v.ukc_m ?? '—'} m</strong>
            </div>
          </div>
        `).join('');

        data.vessels.forEach(v => {
          const color = statusColor(v.status);
          if (!markers[v.vessel_id]) {
            markers[v.vessel_id] = L.circleMarker([v.lat, v.lon], { radius: 9, color: '#0f172a', weight: 2, fillColor: color, fillOpacity: 1 })
              .addTo(map).bindTooltip(`${v.name}`);
          } else {
            markers[v.vessel_id].setLatLng([v.lat, v.lon]).setStyle({ fillColor: color });
          }
        });
      });
    }

    refreshWeather();
    setInterval(refreshWeather, 30000);
    setInterval(refreshVessels, 1500);
    refreshVessels();
  </script>
</body>
</html>"""