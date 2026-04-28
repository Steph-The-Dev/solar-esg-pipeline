from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field, field_validator


class PathConfig(BaseModel):
    sentinel_image: str
    mask_image: str
    model_weights: str
    osm_buildings_temp: str

class DataConfig(BaseModel):
    bbox_wgs84: List[float] = Field(..., min_length=4, max_length=4)
    time_range: str
    max_cloud_cover: int
    normalization_factor: float

    @field_validator('bbox_wgs84')
    @classmethod
    def validate_bbox(cls, v):
        # lat_min, lon_min, lat_max, lon_max
        if not (v[0] < v[2] and v[1] < v[3]):
            # This is a basic check, real CRS bounds might vary but min should be less than max
            pass
        return v

class TrainingConfig(BaseModel):
    patch_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    num_patches_per_epoch: int

class InferenceConfig(BaseModel):
    threshold: float
    patch_size: int
    x_shift: float
    y_shift: float
    scale: float

class FullConfig(BaseModel):
    paths: PathConfig
    data: DataConfig
    training: TrainingConfig
    inference: InferenceConfig

def load_config(config_path: str = "config.yaml") -> FullConfig:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return FullConfig(**config_dict)

def get_resolved_config(config_path: str = "config.yaml", root_dir: Path = None) -> FullConfig:
    """Loads config and resolves relative paths against root_dir."""
    config = load_config(config_path)
    if root_dir:
        config.paths.sentinel_image = str(root_dir / config.paths.sentinel_image)
        config.paths.mask_image = str(root_dir / config.paths.mask_image)
        config.paths.model_weights = str(root_dir / config.paths.model_weights)
        config.paths.osm_buildings_temp = str(root_dir / config.paths.osm_buildings_temp)
    return config
