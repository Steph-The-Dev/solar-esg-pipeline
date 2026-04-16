import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio
from rasterio.windows import Window
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SolarRoofDataset(Dataset):
    def __init__(self, image_path, mask_path, patch_size=256, length=100, transform=None, normalization_factor=4000.0):
        """
        Ein PyTorch Dataset für Satellitenbilder. Lazy Loading implementiert!
        Liest die Daten erst im Moment der Batch-Generierung per Window von der Festplatte.
        """
        self.image_path = image_path
        self.mask_path = mask_path
        self.patch_size = patch_size
        self.length = length
        self.transform = transform
        self.normalization_factor = normalization_factor
        
        # Nur Metadaten lesen, keine Pixel in den RAM laden!
        with rasterio.open(self.image_path) as src:
            self.width = src.width
            self.height = src.height
            
        logger.info(f"Dataset initialisiert (Lazy Loading). Dimensionen: {self.width}x{self.height}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Zufälliges Fenster generieren
        top = np.random.randint(0, self.height - self.patch_size)
        left = np.random.randint(0, self.width - self.patch_size)
        window = Window(left, top, self.patch_size, self.patch_size)
        
        # Gezielt nur den benötigten Ausschnitt von der Festplatte laden
        with rasterio.open(self.image_path) as src:
            img_data = src.read(window=window)
            
        with rasterio.open(self.mask_path) as src:
            mask_patch = src.read(1, window=window)
            
        # Radiometrische Kalibrierung
        img_data = img_data.astype(np.float32) / self.normalization_factor
        img_data = np.clip(img_data, 0.0, 1.0)
        
        # Albumentations erwartet (H, W, C)
        img_patch = np.transpose(img_data, (1, 2, 0))
        
        if self.transform:
            augmented = self.transform(image=img_patch, mask=mask_patch)
            img_patch = augmented['image']
            mask_patch = augmented['mask']
            
        return img_patch, mask_patch.long()

if __name__ == "__main__":
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    sat_file = config['paths']['sentinel_image']
    mask_file = config['paths']['mask_image']
    
    training_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2()
    ])
    
    dataset = SolarRoofDataset(
        sat_file, 
        mask_file, 
        patch_size=config['training']['patch_size'], 
        length=5, 
        transform=training_transform,
        normalization_factor=config['data']['normalization_factor']
    )
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    logger.info("--- Tensor Check ---")
    images, labels = next(iter(dataloader))
    
    logger.info(f"Bilder Batch-Shape: {images.shape}")
    logger.info(f"Masken Batch-Shape: {labels.shape}")
    logger.info(f"Bild Wertebereich: Min={images.min():.2f}, Max={images.max():.2f}")
    logger.info("Test erfolgreich beendet.")