import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
import tempfile
from pathlib import Path

@pytest.fixture
def dummy_data_paths():
    """Erstellt temporäre Dummy-GeoTIFFs für Tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        sat_path = tmp_path / "dummy_sat.tif"
        mask_path = tmp_path / "dummy_mask.tif"
        
        # Erstelle 4-Kanal Dummy Satellitenbild (100x100)
        res = 10
        transform = from_origin(7.5, 47.5, res, res)
        
        with rasterio.open(
            sat_path, 'w', driver='GTiff',
            height=100, width=100, count=4, dtype='uint16',
            crs='EPSG:4326', transform=transform
        ) as dst:
            for i in range(1, 5):
                dst.write(np.full((100, 100), i * 1000, dtype='uint16'), i)
                
        # Erstelle 1-Kanal Dummy Maske (100x100)
        with rasterio.open(
            mask_path, 'w', driver='GTiff',
            height=100, width=100, count=1, dtype='uint8',
            crs='EPSG:4326', transform=transform
        ) as dst:
            mask_data = np.zeros((100, 100), dtype='uint8')
            mask_data[10:30, 10:30] = 1 # Simuliere ein Gebäude
            dst.write(mask_data, 1)
            
        yield sat_path, mask_path
