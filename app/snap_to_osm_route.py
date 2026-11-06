from pathlib import Path
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, substring

GEOJSON_DIR = Path("data/geojson")
SHP_DIR = Path("data/shapefiles")
UTM31N = "EPSG:32631"
WGS84 = "EPSG:4326"

# Terminal coordinates (lon, lat)
IKORODU_PT = Point(3.4880, 6.6010)
FALOMO_PT = Point(3.4273, 6.4425)


def to_single_linestring(geom) -> LineString:
    if isinstance(geom, LineString):
        return geom
    if isinstance(geom, MultiLineString):
        merged = linemerge(geom)
        if isinstance(merged, LineString):
            return merged
        # If still disjoint, concatenate coordinates in order
        coords = []
        for part in merged.geoms:
            coords.extend(list(part.coords))
        return LineString(coords)
    raise ValueError(f"Unsupported geometry: {geom.geom_type}")


def main():
    routes_gdf = gpd.read_file(GEOJSON_DIR / "lagos_ferry_routes.geojson").to_crs(UTM31N)
    ikorodu_utm = gpd.GeoSeries([IKORODU_PT], crs=WGS84).to_crs(UTM31N).iloc[0]
    falomo_utm = gpd.GeoSeries([FALOMO_PT], crs=WGS84).to_crs(UTM31N).iloc[0]

    # Score every OSM ferry route by how close it passes to BOTH Ikorodu and Falomo
    candidates = []
    for idx, row in routes_gdf.iterrows():
        line = to_single_linestring(row.geometry)
        d_iko = line.distance(ikorodu_utm)
        d_fal = line.distance(falomo_utm)
        candidates.append((d_iko + d_fal, d_iko, d_fal, idx, row.get("name"), line))

    candidates.sort(key=lambda x: x[0])
    best_total, d_iko, d_fal, best_idx, best_name, best_line = candidates[0]
    print(f"Best matching OSM route: #{best_idx} ({best_name})")
    print(f"  Distance to Ikorodu jetty: {d_iko:.1f} m")
    print(f"  Distance to Falomo jetty:  {d_fal:.1f} m")

    if d_iko < 1500 and d_fal < 1500:
        # Trim the OSM route so it starts at Ikorodu and ends at Falomo
        proj_iko = best_line.project(ikorodu_utm)
        proj_fal = best_line.project(falomo_utm)
        if proj_iko > proj_fal:
            # Reverse line so direction is always Ikorodu (0%) -> Falomo (100%)
            best_line = LineString(list(best_line.coords)[::-1])
            proj_iko = best_line.project(ikorodu_utm)
            proj_fal = best_line.project(falomo_utm)

        route_utm = substring(best_line, proj_iko, proj_fal)
        out_gdf = gpd.GeoDataFrame(
            [{"name": best_name or "Ikorodu - Falomo (OSM Verified)", "from": "Ikorodu", "to": "Falomo"}],
            geometry=[route_utm],
            crs=UTM31N,
        ).to_crs(WGS84)
    else:
        print("No single OSM route spanned both terminals within 1.5km; using water-routed fallback.")
        # High-density water-only centerline following the OSM ferry channel around Ibeshe,
        # East of Banana Island, under Lekki-Ikoyi Bridge, and through Five Cowries Creek
        water_coords = [
            (3.4880, 6.6010),  # Ebute Ikorodu Ferry Terminal
            (3.4835, 6.5975),  # Clear Ikorodu jetty westward into deeper water
            (3.4745, 6.5840),  # West of Ibeshe headland (stays in open lagoon)
            (3.4650, 6.5650),  # West of Ibeshe beach
            (3.4585, 6.5450),  # Open Lagoon mid-north
            (3.4550, 6.5180),  # Open Lagoon center
            (3.4610, 6.4920),  # Approach north of Banana Island channel
            (3.4695, 6.4750),  # North-East of Banana Island (in the channel, NOT on the island)
            (3.4715, 6.4635),  # East of Banana Island
            (3.4630, 6.4535),  # Approach to Lekki-Ikoyi Link Bridge
            (3.4558, 6.4495),  # Under Lekki-Ikoyi Link Bridge navigation span
            (3.4525, 6.4445),  # Five Cowries Creek East bend
            (3.4510, 6.4375),  # Five Cowries Creek South bend (off Lekki Phase 1)
            (3.4450, 6.4355),  # Five Cowries Creek bottom turn
            (3.4385, 6.4395),  # Five Cowries Creek mid-channel
            (3.4320, 6.4415),  # Five Cowries Creek approaching Falomo
            (3.4273, 6.4425),  # Falomo / Five Cowries Terminal
        ]
        out_gdf = gpd.GeoDataFrame(
            [{"name": "Ikorodu - Falomo Water Channel", "from": "Ikorodu", "to": "Falomo"}],
            geometry=[LineString(water_coords)],
            crs=WGS84,
        )

    out_gdf.to_file(GEOJSON_DIR / "pilot_corridor_route.geojson", driver="GeoJSON")
    out_gdf.to_file(SHP_DIR / "pilot_corridor_route.shp")
    print("Updated pilot_corridor_route.geojson and .shp successfully!")


if __name__ == "__main__":
    main()