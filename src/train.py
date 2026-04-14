import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

# Importiere unser Dataset aus der vorherigen Datei
from dataset import SolarRoofDataset

def train_model():
    print("🚀 Initialisiere Training-Pipeline...")
    
    # 1. Hardware Check (Nutzen wir deine Nvidia-GPU oder die CPU?)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Nutze Recheneinheit: {device}")
    
    # 2. Hyperparameter (Die Stellschrauben für unser PoC-Training)
    BATCH_SIZE = 8
    EPOCHS = 10  # Für den Proof of Concept reichen erstmal 10 Durchläufe
    LEARNING_RATE = 0.001
    
    # 3. Daten vorbereiten
    sat_file = "../data/raw/basel_sentinel2_cropped.tif"
    mask_file = "../data/processed/basel_roof_mask.tif"
    
    # Gleiche Augmentierung wie beim Sanity Check
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2()
    ])
    
    # Wir generieren 200 zufällige Ausschnitte (Patches) pro Epoche
    dataset = SolarRoofDataset(sat_file, mask_file, patch_size=64, length=200, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 4. Das Modell (U-Net Architektur mit einem ResNet34 "Gehirn" als Basis)
    print("🏗️ Baue U-Net Modell...")
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights=None,      # Wir starten mit einem unbeschriebenen Blatt
        in_channels=4,             # Wir haben 4 Bänder (RGB + Nahes Infrarot)
        classes=1                  # Wir wollen 1 Output: Dach oder Nicht-Dach?
    ).to(device)
    
    # 5. Loss & Optimizer
    # DiceLoss ist perfekt für extrem unbalancierte Daten (wie Dächer in einer großen Stadt)
    loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 6. Der Trainings-Loop
    print(f"🔥 Starte Training für {EPOCHS} Epochen...")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            # Schiebe die Daten auf die Grafikkarte (falls vorhanden)
            images = images.to(device)
            
            # SMP erwartet die Masken oft als Float-Tensor mit einer Channel-Dimension
            masks = masks.to(device).float().unsqueeze(1) 
            
            # a) Vorwärtsdurchlauf (Das Modell rät, wo Dächer sind)
            predictions = model(images)
            
            # b) Fehlerberechnung (Wie weit lag das Modell daneben?)
            loss = loss_fn(predictions, masks)
            
            # c) Rückwärtsdurchlauf (Lernen aus Fehlern)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Durchschnittlicher Fehler in dieser Epoche
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoche [{epoch+1}/{EPOCHS}] - Dice Loss: {avg_loss:.4f} " + ("📉 (Verbessert!)" if epoch == 0 or avg_loss < prev_loss else ""))
        prev_loss = avg_loss
        
    # 7. Modell speichern
    os.makedirs("../models", exist_ok=True)
    save_path = "../models/solar_unet_poc.pth"
    torch.save(model.state_dict(), save_path)
    print(f"💾 Training abgeschlossen. Modellgewichte gespeichert unter: {save_path}")

if __name__ == "__main__":
    train_model()