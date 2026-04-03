import requests
import geopandas as gpd
import os

def download_osm_buildings(bbox, output_path):
    """
    Lädt Gebäude-Footprints via Overpass API herunter und speichert sie als GeoJSON.
    
    bbox: Tuple (Süd, West, Nord, Ost) - Bounding Box Koordinaten
    output_path: Pfad zum Speichern der Datei
    """
    print(f"Starte Download der OSM-Daten für BBox: {bbox}...")
    
    # Die Overpass QL (Query Language) Abfrage
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json][timeout:50];
    (
      way["building"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
      relation["building"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    );
    out body;
    >;
    out skel qt;
    """
    
    response = requests.post(overpass_url, data={'data': overpass_query})
    
    if response.status_code == 200:
        data = response.json()
        
        # Geopandas kann die Overpass JSON nicht direkt lesen, wir nutzen einen kleinen Trick
        # indem wir die Elemente in ein sauberes GeoDataFrame parsen
        print("Daten empfangen. Konvertiere in räumliches Format...")
        
        # Einfachster Weg für OSM zu GeoJSON in Python: osmnx (wir nutzen hier aber
        # einen API-Direktaufruf, um Abhängigkeiten schlank zu halten).
        # Speichern der Rohdaten für den Moment:
        import json
        temp_path = output_path.replace('.geojson', '_raw.json')
        with open(temp_path, 'w') as f:
            json.dump(data, f)
            
        print(f"Rohdaten gespeichert unter: {temp_path}")
        print("Tipp für später: Für komplexe topologische Konvertierungen integrieren wir später 'osmnx'.")
    else:
        print(f"Fehler beim Download: Status Code {response.status_code}")

if __name__ == "__main__":
    # Eine kleine Bounding Box im Herzen des Clusters (Richtung Rhein/Roche Tower als Beispiel)
    # Format: (Süd, West, Nord, Ost)
    basel_bbox = (47.5550, 7.5900, 47.5650, 7.6050)
    
    # Stelle sicher, dass der Zielordner existiert
    os.makedirs("../data/raw", exist_ok=True)
    out_file = "../data/raw/basel_buildings.geojson"
    
    download_osm_buildings(basel_bbox, out_file)