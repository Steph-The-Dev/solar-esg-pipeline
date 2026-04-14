import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SolarRoofDataset(Dataset):
    def __init__(self, image_path, mask_path, patch_size=256, length=100, transform=None):
        """
        Ein PyTorch Dataset für Satellitenbilder.
        Liest die Daten ein und generiert zufällige "Patches" (Bildausschnitte).
        """
        print("Lade Daten in den Arbeitsspeicher für schnelles Training...")
        
        # 1. Bild und Maske laden
        with rasterio.open(image_path) as src:
            # Wir lesen alle 4 Bänder (Blau, Grün, Rot, Nahes Infrarot)
            # Shape ist hier (Channels, Height, Width)
            img_data = src.read() 
            
        with rasterio.open(mask_path) as src:
            self.mask = src.read(1) # Shape: (Height, Width)
            
        # 2. Radiometrische Kalibrierung (EXTREM WICHTIG!)
        # Sentinel-2 Pixelwerte gehen oft von 0 bis 10000. 
        # Neuronale Netze hassen große Zahlen. Wir müssen sie auf 0.0 bis 1.0 normalisieren.
        # Ein Standard-Trick in der Fernerkundung: Wir teilen durch 4000 und schneiden Ausreißer ab.
        img_data = img_data.astype(np.float32) / 4000.0
        img_data = np.clip(img_data, 0.0, 1.0)
        
        # Albumentations erwartet das Format (Height, Width, Channels), also sortieren wir um
        self.image = np.transpose(img_data, (1, 2, 0))
        
        self.patch_size = patch_size
        self.length = length # Wie viele Patches wollen wir pro "Epoche" trainieren?
        self.transform = transform

    def __len__(self):
        # Sagt PyTorch, wie groß unser Datensatz virtuell ist
        return self.length

    def __getitem__(self, idx):
        # 3. Der Patch-Generator
        # Wir suchen uns eine zufällige X/Y Koordinate, die weit genug vom Rand weg ist
        h, w, _ = self.image.shape
        top = np.random.randint(0, h - self.patch_size)
        left = np.random.randint(0, w - self.patch_size)
        
        # Wir schneiden exakt an dieser Stelle unser 256x256 Quadrat aus Bild und Maske
        img_patch = self.image[top:top+self.patch_size, left:left+self.patch_size, :]
        mask_patch = self.mask[top:top+self.patch_size, left:left+self.patch_size]
        
        # 4. Augmentierung (Drehen, Spiegeln) & PyTorch-Konvertierung
        if self.transform:
            augmented = self.transform(image=img_patch, mask=mask_patch)
            img_patch = augmented['image']
            mask_patch = augmented['mask']
            
        # Wir geben das fertige Bild und das Label als PyTorch-Tensoren zurück
        return img_patch, mask_patch.long()

# --- TEST-BLOCK ---
if __name__ == "__main__":
    sat_file = "../data/raw/basel_sentinel2_cropped.tif"
    mask_file = "../data/processed/basel_roof_mask.tif"
    
    # Wir definieren unsere Augmentierungs-Pipeline
    training_transform = A.Compose([
        A.HorizontalFlip(p=0.5), # 50% Chance, das Bild horizontal zu spiegeln
        A.VerticalFlip(p=0.5),   # 50% Chance auf vertikale Spiegelung
        A.RandomRotate90(p=0.5),
        ToTensorV2()             # Wandelt die Numpy-Arrays in PyTorch-Tensoren um
    ])
    
    # Dataset instanziieren
    dataset = SolarRoofDataset(sat_file, mask_file, patch_size=64, length=5, transform=training_transform)
    
    # DataLoader bauen (der füttert später das Netz mit 'Batches', z.B. 4 Bilder gleichzeitig)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Test-Abruf
    print("\n--- Tensor Check ---")
    images, labels = next(iter(dataloader))
    
    print(f"Bilder Batch-Shape: {images.shape} -> (Batch-Size, Channels, Height, Width)")
    print(f"Masken Batch-Shape: {labels.shape} -> (Batch-Size, Height, Width)")
    print(f"Bild Wertebereich: Min={images.min():.2f}, Max={images.max():.2f}")
    
    print("\n✅ Wenn diese Tensor-Dimensionen gedruckt wurden, ist deine Daten-Pipeline Deep-Learning-ready!")