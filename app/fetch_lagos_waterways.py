from pathlib import Path
import geopandas as gpd
import osmnx as ox
from shapely.geometry import LineString, Point

# Bounding box covering Lagos Harbour, Five Cowries, Lagoon, up to Ikorodu
# Format for osmnx (v2+ uses (west, south, east, north); we handle both v1 and v2 below)
WEST, SOUTH, EAST, NORTH = 3.33, 6.40, 3.55, 6.63

DATA_DIR = Path("data")
SHP_DIR = DATA_DIR / "shapefiles"
GEOJSON_DIR = DATA_DIR / "geojson"
SHP_DIR.mkdir(parents=True, exist_ok=True)
GEOJSON_DIR.mkdir(parents=True, exist_ok=True)


def fetch_osm_features(tags: dict) -> gpd.GeoDataFrame:
    """Compatible wrapper for osmnx v1.x and v2.x."""
    try:
        # osmnx >= 2.0 expects bbox=(west, south, east, north)
        return ox.features_from_bbox(bbox=(WEST, SOUTH, EAST, NORTH), tags=tags)
    except TypeError:
        # osmnx 1.x expects (north, south, east, west)
        return ox.features_from_bbox(NORTH, SOUTH, EAST, WEST, tags=tags)


def clean_for_shapefile(gdf: gpd.GeoDataFrame, keep_cols: list[str]) -> gpd.GeoDataFrame:
    """Keep only simple string/numeric columns and valid geometries so .shp export succeeds."""
    available = [c for c in keep_cols if c in gdf.columns]
    out = gdf[available + ["geometry"]].copy()
    for col in available:
        out[col] = out[col].astype(str).str.slice(0, 250)
    return out.reset_index(drop=True)


def main():
    print("1/3 Fetching Lagos Lagoon & inland water polygons from OSM...")
    water_raw = fetch_osm_features({"natural": ["water", "bay"], "waterway": ["riverbank", "canal"]})
    water_polys = water_raw[water_raw.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    water_polys = clean_for_shapefile(water_polys, ["name", "natural", "water"])
    water_polys.to_file(SHP_DIR / "lagos_waterways.shp")
    water_polys.to_file(GEOJSON_DIR / "lagos_waterways.geojson", driver="GeoJSON")
    print(f"    Saved {len(water_polys)} water polygons.")

    print("2/3 Fetching mapped ferry routes...")
    routes_raw = fetch_osm_features({"route": "ferry"})
    routes_lines = routes_raw[routes_raw.geometry.type.isin(["LineString", "MultiLineString"])].copy()
    routes_lines = clean_for_shapefile(routes_lines, ["name", "from", "to", "operator"])

    # Also save the specific Ikorodu -> Falomo pilot route anchors as a guaranteed reference line
    pilot_coords = [
        (3.48818, 6.60086),  # Ikorodu (Ipakodo)
        (3.48000, 6.57500),
        (3.46800, 6.54000),
        (3.45800, 6.50500),
        (3.45000, 6.46700),  # East of Banana Island
        (3.45580, 6.45020),  # Lekki-Ikoyi Link Bridge
        (3.43800, 6.44350),  # Five Cowries Creek
        (3.42725, 6.44248),  # Falomo (Five Cowries Terminal)
    ]
    pilot_gdf = gpd.GeoDataFrame(
        [{"name": "Ikorodu - Falomo Pilot Corridor", "from": "Ikorodu", "to": "Falomo"}],
        geometry=[LineString(pilot_coords)],
        crs="EPSG:4326",
    )

    if len(routes_lines) > 0:
        routes_lines.to_file(SHP_DIR / "lagos_ferry_routes.shp")
        routes_lines.to_file(GEOJSON_DIR / "lagos_ferry_routes.geojson", driver="GeoJSON")
    pilot_gdf.to_file(SHP_DIR / "pilot_corridor_route.shp")
    pilot_gdf.to_file(GEOJSON_DIR / "pilot_corridor_route.geojson", driver="GeoJSON")
    print(f"    Saved {len(routes_lines)} OSM ferry routes + 1 Pilot Corridor centerline.")

    print("3/3 Fetching ferry terminals / jetties...")
    terminals_raw = fetch_osm_features({"amenity": "ferry_terminal"})
    terminals_pts = terminals_raw.copy()
    # Convert any terminal polygons to their centroid point
    terminals_pts["geometry"] = terminals_pts.to_crs("EPSG:32631").geometry.centroid.to_crs("EPSG:4326")
    terminals_pts = clean_for_shapefile(terminals_pts, ["name", "operator"])
    terminals_pts.to_file(SHP_DIR / "lagos_ferry_terminals.shp")
    terminals_pts.to_file(GEOJSON_DIR / "lagos_ferry_terminals.geojson", driver="GeoJSON")
    print(f"    Saved {len(terminals_pts)} ferry terminals.")
    print("\nDone! Check data/shapefiles/ and data/geojson/.")


if __name__ == "__main__":
    main()