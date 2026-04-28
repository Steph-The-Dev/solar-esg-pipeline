import torch
import numpy as np
from src.dataset import SolarRoofDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def test_dataset_initialization(dummy_data_paths):
    sat_path, mask_path = dummy_data_paths
    dataset = SolarRoofDataset(str(sat_path), str(mask_path), patch_size=32, length=10)
    
    assert len(dataset) == 10
    assert dataset.width == 100
    assert dataset.height == 100

def test_dataset_getitem(dummy_data_paths):
    sat_path, mask_path = dummy_data_paths
    patch_size = 32
    dataset = SolarRoofDataset(str(sat_path), str(mask_path), patch_size=patch_size, length=5)
    
    img, mask = dataset[0]
    
    # Check types
    assert isinstance(img, np.ndarray)
    assert isinstance(mask, np.ndarray)
    
    # Check shapes (H, W, C for img before transform, but here no transform so it stays H, W, C or C, H, W depending on dataset)
    # Looking at dataset.py: 
    # img_patch = np.transpose(img_data, (1, 2, 0)) -> (H, W, C)
    # return img_patch, mask_patch.long()
    
    assert img.shape == (patch_size, patch_size, 4)
    assert mask.shape == (patch_size, patch_size)
    
    # Check normalization
    assert img.max() <= 1.0
    assert img.min() >= 0.0

def test_dataset_with_transform(dummy_data_paths):
    sat_path, mask_path = dummy_data_paths
    patch_size = 32
    
    transform = A.Compose([
        ToTensorV2()
    ])
    
    dataset = SolarRoofDataset(
        str(sat_path), 
        str(mask_path), 
        patch_size=patch_size, 
        length=5, 
        transform=transform
    )
    
    img, mask = dataset[0]
    
    # After ToTensorV2, it should be a torch.Tensor and (C, H, W)
    assert torch.is_tensor(img)
    assert torch.is_tensor(mask)
    assert img.shape == (4, patch_size, patch_size)
    assert mask.shape == (patch_size, patch_size)
