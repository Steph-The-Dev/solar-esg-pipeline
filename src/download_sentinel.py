import logging
import os
from pathlib import Path

import geopandas as gpd
import rasterio
import requests
from rasterio.windows import from_bounds
from shapely.geometry import box

from src.config_schema import FullConfig, get_resolved_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_sentinel2_cog(
    bbox_wgs84: list[float], output_path: str, time_range: str, max_cloud_cover: int
) -> None:
    """
    Searches for and downloads Sentinel-2 cloud-free imagery via the STAC API.
    Streams and crops pixel data directly from the cloud (COG).

    Args:
        bbox_wgs84: Bounding box in WGS84 coordinates [lat_min, lon_min, lat_max, lon_max].
        output_path: Path to save the resulting 4-band GeoTIFF.
        time_range: Datetime range for the search (e.g., "2023-01-01/2023-12-31").
        max_cloud_cover: Maximum allowed cloud cover percentage.
    """
    logger.info("Searching for cloud-free Sentinel-2 images via STAC API...")

    url = "https://earth-search.aws.element84.com/v1/search"
    # Convert to STAC bbox format [min_lon, min_lat, max_lon, max_lat]
    stac_bbox = [bbox_wgs84[1], bbox_wgs84[0], bbox_wgs84[3], bbox_wgs84[2]]

    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": stac_bbox,
        "datetime": time_range,
        "query": {"eo:cloud_cover": {"lt": max_cloud_cover}},
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        "limit": 1,
    }

    response = requests.post(url, json=payload)

    if response.status_code != 200:
        logger.error(f"API ERROR ({response.status_code}): {response.text}")
        return

    items = response.json().get("features", [])

    if not items:
        logger.warning(
            f"No images with less than {max_cloud_cover}% cloud cover "
            f"found in period {time_range}."
        )
        return

    item = items[0]
    logger.info(f"Image found! Acquisition date: {item['properties']['datetime']}")
    logger.info(f"Cloud cover: {item['properties']['eo:cloud_cover']}%")

    geom = box(*stac_bbox)
    gdf_wgs84 = gpd.GeoDataFrame({"geometry": [geom]}, crs="EPSG:4326")

    blue_url = item["assets"]["blue"]["href"]
    green_url = item["assets"]["green"]["href"]
    red_url = item["assets"]["red"]["href"]
    nir_url = item["assets"]["nir"]["href"]

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Streaming and cropping pixel data from COG...")

    with rasterio.open(blue_url) as src:
        sat_crs = src.crs
        gdf_utm = gdf_wgs84.to_crs(sat_crs)
        utm_bounds = gdf_utm.total_bounds

        window = from_bounds(*utm_bounds, transform=src.transform)

        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": int(window.height),
                "width": int(window.width),
                "transform": rasterio.windows.transform(window, src.transform),
                "count": 4,
            }
        )

        with rasterio.open(output_path, "w", **out_meta) as dest:
            for idx, url_band in enumerate([blue_url, green_url, red_url, nir_url], start=1):
                logger.info(f"Downloading Band {idx}/4...")
                with rasterio.open(url_band) as band_src:
                    cropped_data = band_src.read(1, window=window)
                    dest.write(cropped_data, idx)

    logger.info(f"Data saved to: {output_path}")


if __name__ == "__main__":
    SRC_DIR = Path(__file__).resolve().parent
    ROOT_DIR = SRC_DIR.parent
    config_path = ROOT_DIR / "config.yaml"

    config: FullConfig = get_resolved_config(str(config_path), ROOT_DIR)

    download_sentinel2_cog(
        bbox_wgs84=config.data.bbox_wgs84,
        output_path=config.paths.sentinel_image,
        time_range=config.data.time_range,
        max_cloud_cover=config.data.max_cloud_cover,
    )
