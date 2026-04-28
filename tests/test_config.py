import yaml
from pathlib import Path
import pytest

def test_config_structure():
    config_path = Path("config.yaml")
    assert config_path.exists()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    required_keys = ['paths', 'data', 'training', 'inference']
    for key in required_keys:
        assert key in config, f"Missing required key: {key}"
        
    # Check specific paths
    assert 'sentinel_image' in config['paths']
    assert 'mask_image' in config['paths']
    assert 'model_weights' in config['paths']
    
    # Check data parameters
    assert 'bbox_wgs84' in config['data']
    assert len(config['data']['bbox_wgs84']) == 4
    
    # Check training parameters
    assert 'patch_size' in config['training']
    assert 'batch_size' in config['training']
    assert 'epochs' in config['training']
