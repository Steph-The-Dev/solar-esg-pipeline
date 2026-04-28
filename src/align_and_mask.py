import logging
import os
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import rasterio
import rasterio.features
from shapely.geometry import box

from src.config_schema import FullConfig, get_resolved_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_alignment_mask(sentinel_path: str, output_mask_path: str) -> None:
    """
    Creates a binary mask of buildings from OSM data, aligned with a reference satellite image.

    Args:
        sentinel_path: Path to the reference Sentinel-2 GeoTIFF.
        output_mask_path: Path to save the resulting binary mask GeoTIFF.
    """
    logger.info(f"Reading reference metadata from: {sentinel_path}")

    with rasterio.open(sentinel_path) as src:
        sat_crs = src.crs
        sat_transform = src.transform
        sat_width = src.width
        sat_height = src.height
        sat_bounds = src.bounds

        geom = box(sat_bounds.left, sat_bounds.bottom, sat_bounds.right, sat_bounds.top)
        bbox_gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs=sat_crs)
        bbox_wgs84 = bbox_gdf.to_crs("EPSG:4326").total_bounds

    logger.info("Downloading building polygons via OSMnx for the satellite window...")

    bbox_tuple = tuple(bbox_wgs84)
    tags = {"building": True}
    buildings = ox.features_from_bbox(bbox=bbox_tuple, tags=tags)

    buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]
    logger.info(f"{len(buildings)} building polygons found.")

    logger.info("Reprojecting buildings to match satellite raster (CRS Alignment)...")
    buildings_utm = buildings.to_crs(sat_crs)

    logger.info("Rasterizing vectors...")
    shapes = ((geom, 1) for geom in buildings_utm.geometry)

    mask = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=(sat_height, sat_width),
        transform=sat_transform,
        fill=0,
        all_touched=True,
        dtype=rasterio.uint8,
    )

    logger.info("Saving the final label mask...")
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff", "count": 1, "dtype": rasterio.uint8})

    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    with rasterio.open(output_mask_path, "w", **out_meta) as dest:
        dest.write(mask, 1)

    logger.info(f"Mask successfully created at: {output_mask_path}")


if __name__ == "__main__":
    SRC_DIR = Path(__file__).resolve().parent
    ROOT_DIR = SRC_DIR.parent
    config_path = ROOT_DIR / "config.yaml"

    config: FullConfig = get_resolved_config(str(config_path), ROOT_DIR)

    create_alignment_mask(config.paths.sentinel_image, config.paths.mask_image)
