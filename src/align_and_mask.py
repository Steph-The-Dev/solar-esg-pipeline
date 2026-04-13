import rasterio
import rasterio.features
import geopandas as gpd
import osmnx as ox
import numpy as np
import cv2
import os
from shapely.geometry import box

def create_alignment_mask(sentinel_path, output_mask_path):
    print(f"Lese Referenz-Metadaten aus: {sentinel_path}")
    
    with rasterio.open(sentinel_path) as src:
        # 1. Metadaten des Satellitenbildes auslesen
        sat_crs = src.crs
        sat_transform = src.transform
        sat_width = src.width
        sat_height = src.height
        sat_bounds = src.bounds # Das sind die Grenzen im UTM-Format
        
        # 2. Bounding Box in GPS-Koordinaten (WGS84) umwandeln für OSM
        # Wir nutzen Shapely, um aus den Satelliten-Grenzen ein sauberes Rechteck zu zeichnen
        geom = box(sat_bounds.left, sat_bounds.bottom, sat_bounds.right, sat_bounds.top)
        
        # Daraus machen wir ein GeoDataFrame
        bbox_gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs=sat_crs)
        bbox_wgs84 = bbox_gdf.to_crs("EPSG:4326").total_bounds # [West, Süd, Ost, Nord]
        
    print(f"Lade Gebäude-Polygone via OSMnx exakt für das Satelliten-Fenster...")
    
    # OSMnx 2.0 Update: Wir übergeben die Box als ein einziges Tupel
    # Format: (Left, Bottom, Right, Top) - exakt das, was GeoPandas uns liefert!
    bbox_tuple = tuple(bbox_wgs84)
    
    tags = {'building': True}
    buildings = ox.features_from_bbox(bbox=bbox_tuple, tags=tags)
    
    # Filtern: Wir behalten nur echte Polygone (keine einzelnen Punkte)
    buildings = buildings[buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    print(f"✅ {len(buildings)} Gebäude-Polygone gefunden.")
    
    print("Reprojiziere Gebäude auf das Satelliten-Raster (CRS-Alignment)...")
    buildings_utm = buildings.to_crs(sat_crs)
    
    print("Rasterisiere Vektoren (OpenCV & Rasterio Magie)...")
    # Wir erstellen ein leeres Bild (nur Nullen = Schwarz) in der Größe des Satellitenbildes
    mask = np.zeros((sat_height, sat_width), dtype=np.uint8)
    
    # Extrahieren der Geometrien als Liste
    shapes = ((geom, 1) for geom in buildings_utm.geometry)
    
    # Rasterio 'brennt' die Vektor-Polygone in unser leeres Pixelraster (1 = Weiß)
    mask = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=(sat_height, sat_width),
        transform=sat_transform,
        fill=0,
        all_touched=True,
        dtype=rasterio.uint8
    )
    
    # Optional: OpenCV nutzen wir hier für Morphologische Operationen.
    # Da Dächer in Satellitenbildern oft leicht überstehen, können wir die Maske leicht vergrößern (Dilatation)
    # kernel = np.ones((3,3), np.uint8)
    # mask = cv2.dilate(mask, kernel, iterations=1)
    
    print("Speichere die fertige Label-Maske...")
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "count": 1,        # Nur noch 1 Band (Schwarz/Weiß)
        "dtype": rasterio.uint8
    })
    
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    with rasterio.open(output_mask_path, "w", **out_meta) as dest:
        dest.write(mask, 1)
        
    print(f"🎉 Maske erfolgreich erstellt unter: {output_mask_path}")

if __name__ == "__main__":
    sat_file = "../data/raw/basel_sentinel2_cropped.tif"
    mask_file = "../data/processed/basel_roof_mask.tif"
    
    create_alignment_mask(sat_file, mask_file)