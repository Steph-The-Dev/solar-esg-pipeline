import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import albumentations as alb
import numpy as np
import rasterio
import torch
from albumentations.pytorch import ToTensorV2
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset

from src.config_schema import FullConfig, get_resolved_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SolarRoofDataset(Dataset):
    """
    A PyTorch Dataset for satellite imagery.
    Implements Lazy Loading to read data from disk only when a batch is generated.
    """

    def __init__(
        self,
        image_path: str,
        mask_path: str,
        patch_size: int = 256,
        length: int = 100,
        transform: Optional[alb.Compose] = None,
        normalization_factor: float = 4000.0,
    ):
        """
        Initializes the dataset metadata without loading pixel data into memory.

        Args:
            image_path: Path to the Sentinel-2 GeoTIFF.
            mask_path: Path to the binary mask GeoTIFF.
            patch_size: Size of the random patches to extract.
            length: Number of patches to return per epoch.
            transform: Albumentations transformation pipeline.
            normalization_factor: Factor to normalize pixel values.
        """
        self.image_path = image_path
        self.mask_path = mask_path
        self.patch_size = patch_size
        self.length = length
        self.transform = transform
        self.normalization_factor = normalization_factor

        with rasterio.open(self.image_path) as src:
            self.width = src.width
            self.height = src.height

        logger.info(f"Dataset initialized (Lazy Loading). Dimensions: {self.width}x{self.height}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(
        self, idx: int
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Generates a random window and loads the corresponding data from disk.
        """
        top = np.random.randint(0, self.height - self.patch_size)
        left = np.random.randint(0, self.width - self.patch_size)
        window = Window(left, top, self.patch_size, self.patch_size)

        with rasterio.open(self.image_path) as src:
            img_data = src.read(window=window)

        with rasterio.open(self.mask_path) as src:
            mask_patch = src.read(1, window=window)

        # Radiometric calibration
        img_data = img_data.astype(np.float32) / self.normalization_factor
        img_data = np.clip(img_data, 0.0, 1.0)

        # Albumentations expects (H, W, C)
        img_patch = np.transpose(img_data, (1, 2, 0))

        if self.transform:
            augmented = self.transform(image=img_patch, mask=mask_patch)
            img_patch = augmented["image"]
            mask_patch = augmented["mask"]

        if torch.is_tensor(mask_patch):
            return img_patch, mask_patch.long()
        else:
            return img_patch, mask_patch.astype(np.int64)


if __name__ == "__main__":
    SRC_DIR = Path(__file__).resolve().parent
    ROOT_DIR = SRC_DIR.parent
    config_path = ROOT_DIR / "config.yaml"

    config: FullConfig = get_resolved_config(str(config_path), ROOT_DIR)

    training_transform = alb.Compose(
        [
            alb.HorizontalFlip(p=0.5),
            alb.VerticalFlip(p=0.5),
            alb.RandomRotate90(p=0.5),
            ToTensorV2(),
        ]
    )

    dataset = SolarRoofDataset(
        config.paths.sentinel_image,
        config.paths.mask_image,
        patch_size=config.training.patch_size,
        length=5,
        transform=training_transform,
        normalization_factor=config.data.normalization_factor,
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    logger.info("--- Tensor Check ---")
    images, labels = next(iter(dataloader))

    logger.info(f"Image Batch-Shape: {images.shape}")
    logger.info(f"Mask Batch-Shape: {labels.shape}")
    logger.info(f"Image Value Range: Min={images.min():.2f}, Max={images.max():.2f}")
    logger.info("Test successfully completed.")
