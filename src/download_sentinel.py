import requests
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.windows import from_bounds
import os
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_sentinel2_cog(bbox_wgs84, output_path, time_range, max_cloud_cover):
    logger.info("Suche via STAC API nach wolkenfreien Sentinel-2 Bildern...")
    
    url = "https://earth-search.aws.element84.com/v1/search"
    stac_bbox = [bbox_wgs84[1], bbox_wgs84[0], bbox_wgs84[3], bbox_wgs84[2]]
    
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": stac_bbox,
        "datetime": time_range,
        "query": {"eo:cloud_cover": {"lt": max_cloud_cover}},
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        "limit": 1
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code != 200:
        logger.error(f"API FEHLER ({response.status_code}): {response.text}")
        return
        
    items = response.json().get("features", [])
    
    if not items:
        logger.warning(f"Keine Bilder mit unter {max_cloud_cover}% Wolkenabdeckung im Zeitraum {time_range} gefunden.")
        return
        
    item = items[0]
    logger.info(f"Perfektes Bild gefunden! Aufnahmedatum: {item['properties']['datetime']}")
    logger.info(f"Wolkenabdeckung: {item['properties']['eo:cloud_cover']}%")
    
    geom = box(*stac_bbox)
    gdf_wgs84 = gpd.GeoDataFrame({"geometry": [geom]}, crs="EPSG:4326")
    
    blue_url = item['assets']['blue']['href']
    green_url = item['assets']['green']['href']
    red_url = item['assets']['red']['href']
    nir_url = item['assets']['nir']['href']
    
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Streame und schneide Pixel-Daten live aus der Cloud (COG)...")
    
    with rasterio.open(blue_url) as src:
        sat_crs = src.crs
        gdf_utm = gdf_wgs84.to_crs(sat_crs)
        utm_bounds = gdf_utm.total_bounds
        
        window = from_bounds(*utm_bounds, transform=src.transform)
        
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": int(window.height),
            "width": int(window.width),
            "transform": rasterio.windows.transform(window, src.transform),
            "count": 4
        })
        
        with rasterio.open(output_path, "w", **out_meta) as dest:
            for idx, url_band in enumerate([blue_url, green_url, red_url, nir_url], start=1):
                logger.info(f"Lade Band {idx}/4 herunter...")
                with rasterio.open(url_band) as band_src:
                    cropped_data = band_src.read(1, window=window)
                    dest.write(cropped_data, idx)
                    
    logger.info(f"Daten gespeichert unter: {output_path}")

if __name__ == "__main__":
    SRC_DIR = Path(__file__).resolve().parent
    ROOT_DIR = SRC_DIR.parent
    config_path = ROOT_DIR / 'config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    download_sentinel2_cog(
        bbox_wgs84=config['data']['bbox_wgs84'],
        output_path=str(ROOT_DIR / config['paths']['sentinel_image']),
        time_range=config['data']['time_range'],
        max_cloud_cover=config['data']['max_cloud_cover']
    )