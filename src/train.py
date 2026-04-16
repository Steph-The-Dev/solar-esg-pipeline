import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import yaml
import logging

from dataset import SolarRoofDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_dataloaders(config):
    sat_file = config['paths']['sentinel_image']
    mask_file = config['paths']['mask_image']
    
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2()
    ])
    
    dataset = SolarRoofDataset(
        sat_file, 
        mask_file, 
        patch_size=config['training']['patch_size'], 
        length=config['training']['num_patches_per_epoch'], 
        transform=train_transform,
        normalization_factor=config['data']['normalization_factor']
    )
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
    return dataloader

def build_model(device):
    logger.info("Baue U-Net Modell...")
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights=None,      
        in_channels=4,             
        classes=1                  
    ).to(device)
    return model

def train_model(config):
    logger.info("Initialisiere Training-Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Nutze Recheneinheit: {device}")
    
    epochs = config['training']['epochs']
    learning_rate = config['training']['learning_rate']
    save_path = config['paths']['model_weights']
    
    dataloader = get_dataloaders(config)
    model = build_model(device)
    
    loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    logger.info(f"Starte Training für {epochs} Epochen...")
    
    prev_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device).float().unsqueeze(1) 
            
            predictions = model(images)
            loss = loss_fn(predictions, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        verbessert = " (Verbessert!)" if avg_loss < prev_loss else ""
        logger.info(f"Epoche [{epoch+1}/{epochs}] - Dice Loss: {avg_loss:.4f}{verbessert}")
        prev_loss = min(prev_loss, avg_loss)
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Training abgeschlossen. Modellgewichte gespeichert unter: {save_path}")

if __name__ == "__main__":
    with open('../config.yaml', 'r') as f:
        config_data = yaml.safe_load(f)
    train_model(config_data)