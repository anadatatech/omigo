from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

DATA_DIR = Path("data")
GEOJSON_DIR = DATA_DIR / "geojson"
SHP_DIR = DATA_DIR / "shapefiles"
TIF_PATH = DATA_DIR / "bathymetry_lagos_poc.tif"

WGS84 = "EPSG:4326"
UTM31N = "EPSG:32631"
NODATA = -9999.0


def build_water_verified_route(water_utm) -> gpd.GeoDataFrame:
    """
    Follows the OSM ferry trajectory through Ikorodu Bay, East of Banana Island,
    and Five Cowries Creek, clamping every 25m vertex inside navigable water.
    """
    raw_wgs = [
        (3.4878, 6.6005),  # Ebute Ikorodu Terminal
        (3.4810, 6.5960),  # Ikorodu channel exit
        (3.4690, 6.5850),  # Ikorodu Bay (west of Ibeshe)
        (3.4575, 6.5670),  # Open Lagoon west of Ibeshe peninsula
        (3.4520, 6.5450),  # Mid-Lagoon North
        (3.4530, 6.5180),  # Mid-Lagoon Center
        (3.4600, 6.4920),  # Approach to Banana Island East channel
        (3.4700, 6.4760),  # North-East of Banana Island
        (3.4712, 6.4630),  # Channel between Banana Island & Lekki
        (3.4680, 6.4555),  # South-East of Banana Island
        (3.4610, 6.4520),  # Entering Five Cowries Creek East
        (3.4558, 6.4495),  # Under Lekki-Ikoyi Link Bridge
        (3.4522, 6.4440),  # Five Cowries Creek East reach
        (3.4508, 6.4375),  # Five Cowries Creek South bend
        (3.4450, 6.4355),  # Five Cowries Creek bottom turn
        (3.4385, 6.4395),  # Five Cowries Creek mid-channel
        (3.4320, 6.4415),  # Five Cowries Creek West reach
        (3.4273, 6.4423),  # Falomo / Five Cowries Terminal
    ]

    line_utm = gpd.GeoSeries([LineString(raw_wgs)], crs=WGS84).to_crs(UTM31N).iloc[0]
    dense_line = line_utm.segmentize(25.0)

    safe_water = water_utm.buffer(-30.0)
    fallback_water = water_utm.buffer(-10.0)

    clamped_coords = []
    for x, y in dense_line.coords:
        pt = Point(x, y)
        if safe_water.contains(pt):
            clamped_coords.append((x, y))
        elif fallback_water.contains(pt):
            nearest_pt = nearest_points(safe_water, pt)[0]
            clamped_coords.append((nearest_pt.x, nearest_pt.y))
        else:
            nearest_pt = nearest_points(fallback_water, pt)[0]
            clamped_coords.append((nearest_pt.x, nearest_pt.y))

    clean_line_utm = LineString(clamped_coords).simplify(8.0)
    out_gdf = gpd.GeoDataFrame(
        [{"name": "Ikorodu - Falomo (OSM Water Verified)", "from": "Ikorodu", "to": "Falomo"}],
        geometry=[clean_line_utm],
        crs=UTM31N,
    ).to_crs(WGS84)

    out_gdf.to_file(GEOJSON_DIR / "pilot_corridor_route.geojson", driver="GeoJSON")
    out_gdf.to_file(SHP_DIR / "pilot_corridor_route.shp")
    print("1/2 Saved water-verified route to data/geojson/pilot_corridor_route.geojson")
    return out_gdf


def main():
    print("Loading full OSM Lagoon Waterway shapefile and 53 OSM ferry routes...")
    water_gdf_wgs = gpd.read_file(GEOJSON_DIR / "lagos_waterways.geojson")
    water_gdf_utm = water_gdf_wgs.to_crs(UTM31N)
    raw_water_utm = water_gdf_utm.geometry.union_all()

    routes_gdf_utm = gpd.read_file(GEOJSON_DIR / "lagos_ferry_routes.geojson").to_crs(UTM31N)
    osm_ferries_utm = routes_gdf_utm.geometry.union_all()

    # Include tidal creek corridors (Five Cowries Creek, Apapa/Marina harbour channels)
    # where OSM ferry lines run through sea-connected channels
    water_utm = raw_water_utm.union(osm_ferries_utm.buffer(120.0))

    # 1. Verify the pilot route inside the complete waterway
    pilot_gdf_wgs = build_water_verified_route(water_utm)
    pilot_utm = pilot_gdf_wgs.to_crs(UTM31N).geometry.iloc[0]
    all_channels_utm = osm_ferries_utm.union(pilot_utm)

    # 2. Use the EXACT total_bounds of the entire waterway shapefile (West to East, South to North)
    full_water_wgs = gpd.GeoSeries([water_utm], crs=UTM31N).to_crs(WGS84).iloc[0]
    minx, miny, maxx, maxy = full_water_wgs.bounds
    pad = 0.003
    west, south, east, north = minx - pad, miny - pad, maxx + pad, maxy + pad

    # Scale raster dimensions proportionally to the full lagoon aspect ratio (~40m resolution)
    width = 1400
    height = int(round(width * ((north - south) / (east - west))))
    transform = from_bounds(west, south, east, north, width, height)
    print(f"    Full shapefile bounds: W={west:.4f}, S={south:.4f}, E={east:.4f}, N={north:.4f} ({width}x{height} px)")

    # Rasterize 100% of the waterway polygon
    water_mask = rasterize(
        [(full_water_wgs, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    # Compute shore-distance depth profile across the ENTIRE shapefile (including Eastern Lagoon to Eleko/Epe)
    shore_steps = [
        (0.0, 0.40),     # Immediate shoreline / bank: 0.40m
        (30.0, 0.80),    # 30m from shore: 0.80m
        (70.0, 1.30),    # 70m from shore: 1.30m
        (130.0, 1.80),   # 130m from shore: 1.80m
        (240.0, 2.25),   # 240m from shore: 2.25m
        (450.0, 2.70),   # Open lagoon body: 2.70m
        (800.0, 3.15),   # Mid-lagoon basin: 3.15m
        (1400.0, 3.60),  # Deep central/eastern lagoon basin: 3.60m
    ]
    depth_grid = np.zeros((height, width), dtype="float32")
    for erode_m, d_val in shore_steps:
        eroded_utm = water_utm.buffer(-erode_m) if erode_m > 0 else water_utm
        if not eroded_utm.is_empty:
            eroded_wgs = gpd.GeoSeries([eroded_utm], crs=UTM31N).to_crs(WGS84).iloc[0]
            mask_step = rasterize([(eroded_wgs, 1)], out_shape=(height, width), transform=transform, fill=0, dtype="uint8")
            depth_grid = np.where(mask_step == 1, d_val, depth_grid)

    # Carve navigable channels along all 53 OSM ferry routes across the whole lagoon
    safe_channel_water_utm = water_utm.buffer(-18.0)
    for buf_m, chan_depth in [(150.0, 2.50), (95.0, 3.15), (50.0, 3.90)]:
        chan_utm = all_channels_utm.buffer(buf_m).intersection(safe_channel_water_utm)
        if not chan_utm.is_empty:
            chan_wgs = gpd.GeoSeries([chan_utm], crs=UTM31N).to_crs(WGS84).iloc[0]
            c_mask = rasterize([(chan_wgs, 1)], out_shape=(height, width), transform=transform, fill=0, dtype="uint8")
            depth_grid = np.where(c_mask == 1, np.maximum(depth_grid, chan_depth), depth_grid)

    # Add natural bathymetric variation (sandbars, shoals, deeper basins) across the full extent
    lons = np.linspace(west, east, width, dtype="float32")
    lats = np.linspace(north, south, height, dtype="float32")
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    undulation = (
        0.35 * np.sin(lon_grid * 180.0) * np.cos(lat_grid * 165.0)
        + 0.25 * np.cos((lon_grid + lat_grid) * 290.0)
    )
    depth_grid = np.clip(depth_grid + undulation, 0.30, 5.40)

    # Known shallow sandbars / mudflats across Western, Central, and Eastern Lagoon
    shallow_flats = [
        (3.473, 6.578, 0.012, 0.75),  # Ibeshe coastal mudflat
        (3.462, 6.602, 0.015, 0.65),  # Majidun / Ikorodu delta siltation
        (3.478, 6.468, 0.010, 0.85),  # East Banana Island / Lekki sandbar
        (3.412, 6.495, 0.016, 0.80),  # Makoko / Yaba shallow flats
        (3.585, 6.515, 0.020, 0.90),  # Baiyeku / Ajah narrows sandbank
        (3.695, 6.555, 0.025, 0.85),  # Eastern Lagoon (Ibeju-Lekki) shoal
    ]
    for slon, slat, rad_deg, min_d in shallow_flats:
        dist_deg = np.hypot(lon_grid - slon, lat_grid - slat)
        factor = np.clip(dist_deg / rad_deg, 0.0, 1.0)
        depth_grid = np.minimum(depth_grid, min_d + (depth_grid - min_d) * (factor ** 1.5))

    final_bathy = np.where(water_mask == 1, np.round(depth_grid, 2), NODATA).astype("float32")

    with rasterio.open(
        TIF_PATH,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=WGS84,
        transform=transform,
        nodata=NODATA,
    ) as dst:
        dst.write(final_bathy, 1)

    print(f"2/2 Saved full-shapefile Bathymetry GeoTIFF: {TIF_PATH}")


if __name__ == "__main__":
    main()