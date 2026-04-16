import rasterio
import rasterio.features
import geopandas as gpd
import osmnx as ox
import numpy as np
import os
import yaml
import logging
from shapely.geometry import box

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_alignment_mask(sentinel_path, output_mask_path):
    logger.info(f"Lese Referenz-Metadaten aus: {sentinel_path}")
    
    with rasterio.open(sentinel_path) as src:
        sat_crs = src.crs
        sat_transform = src.transform
        sat_width = src.width
        sat_height = src.height
        sat_bounds = src.bounds
        
        geom = box(sat_bounds.left, sat_bounds.bottom, sat_bounds.right, sat_bounds.top)
        bbox_gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs=sat_crs)
        bbox_wgs84 = bbox_gdf.to_crs("EPSG:4326").total_bounds
        
    logger.info("Lade Gebäude-Polygone via OSMnx exakt für das Satelliten-Fenster...")
    
    bbox_tuple = tuple(bbox_wgs84)
    tags = {'building': True}
    buildings = ox.features_from_bbox(bbox=bbox_tuple, tags=tags)
    
    buildings = buildings[buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    logger.info(f"{len(buildings)} Gebäude-Polygone gefunden.")
    
    logger.info("Reprojiziere Gebäude auf das Satelliten-Raster (CRS-Alignment)...")
    buildings_utm = buildings.to_crs(sat_crs)
    
    logger.info("Rasterisiere Vektoren...")
    mask = np.zeros((sat_height, sat_width), dtype=np.uint8)
    shapes = ((geom, 1) for geom in buildings_utm.geometry)
    
    mask = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=(sat_height, sat_width),
        transform=sat_transform,
        fill=0,
        all_touched=True,
        dtype=rasterio.uint8
    )
    
    logger.info("Speichere die fertige Label-Maske...")
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "count": 1,
        "dtype": rasterio.uint8
    })
    
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    with rasterio.open(output_mask_path, "w", **out_meta) as dest:
        dest.write(mask, 1)
        
    logger.info(f"Maske erfolgreich erstellt unter: {output_mask_path}")

if __name__ == "__main__":
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    sat_file = config['paths']['sentinel_image']
    mask_file = config['paths']['mask_image']
    
    create_alignment_mask(sat_file, mask_file)