import requests
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.windows import from_bounds
import os

def download_sentinel2_cog(bbox_wgs84, output_dir):
    print("Suche via STAC API nach wolkenfreien Sentinel-2 Bildern...")
    
    url = "https://earth-search.aws.element84.com/v1/search"
    stac_bbox = [bbox_wgs84[1], bbox_wgs84[0], bbox_wgs84[3], bbox_wgs84[2]]
    
    payload = {
        "collections": ["sentinel-2-l2a"],  # Die offizielle, kombinierte Collection für L2A
        "bbox": stac_bbox,
        "datetime": "2023-05-01T00:00:00Z/2023-09-30T23:59:59Z", # Sommer in Basel
        "query": {"eo:cloud_cover": {"lt": 20}},
        # DER FIX: In STAC muss 'properties.' vorangestellt werden!
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        "limit": 1
    }
    
    # Anfrage abschicken
    response = requests.post(url, json=payload)
    
    # Senior-Move: Fehler niemals stumm schlucken!
    if response.status_code != 200:
        print(f"🚨 API FEHLER ({response.status_code}): {response.text}")
        return
        
    items = response.json().get("features", [])
    
    if not items:
        print("API hat korrekt geantwortet, aber unter 20% Wolken in diesem Zeitraum wirklich nichts gefunden.")
        return
        
    item = items[0]
    print(f"✅ Perfektes Bild gefunden! Aufnahmedatum: {item['properties']['datetime']}")
    print(f"☁️ Wolkenabdeckung: {item['properties']['eo:cloud_cover']}%")
    
    geom = box(*stac_bbox)
    gdf_wgs84 = gpd.GeoDataFrame({"geometry": [geom]}, crs="EPSG:4326")
    
    # Wir greifen auf die Cloud-URLs der spezifischen Bänder zu
    blue_url = item['assets']['blue']['href']
    green_url = item['assets']['green']['href']
    red_url = item['assets']['red']['href']
    nir_url = item['assets']['nir']['href']
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "basel_sentinel2_cropped.tif")
    
    print("Streame und schneide Pixel-Daten live aus der Cloud (COG)...")
    
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
        
        with rasterio.open(out_path, "w", **out_meta) as dest:
            for idx, url_band in enumerate([blue_url, green_url, red_url, nir_url], start=1):
                print(f"-> Lade Band {idx}/4 herunter...")
                with rasterio.open(url_band) as band_src:
                    cropped_data = band_src.read(1, window=window)
                    dest.write(cropped_data, idx)
                    
    print(f"🎉 Fertig! Die Daten liegen auf deiner Festplatte unter: {out_path}")

if __name__ == "__main__":
    # Die BBox für das Wettstein-Quartier / Roche Tower
    basel_bbox = (47.5550, 7.5900, 47.5650, 7.6050)
    download_sentinel2_cog(basel_bbox, "../data/raw")