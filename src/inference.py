import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import scipy.ndimage
import logging
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(weights_path, device):
    logger.info("Lade Modell und Gewichte...")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=4,
        classes=1
    ).to(device)
    # map_location ensures we can load GPU weights on CPU if needed
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    return model

def load_and_preprocess_image(image_path, normalization_factor):
    logger.info("Lade Testbild...")
    with rasterio.open(image_path) as src:
        img_data = src.read()
    
    original_shape = img_data.shape
    img_data = img_data.astype(np.float32) / normalization_factor
    img_data = np.clip(img_data, 0.0, 1.0)
    img_t = np.transpose(img_data, (1, 2, 0))
    
    transform = A.Compose([
        A.PadIfNeeded(min_height=128, min_width=128, border_mode=0, fill=0), 
        ToTensorV2()
    ])
    tensor_img = transform(image=img_t)['image'].unsqueeze(0)
    return tensor_img, img_t, original_shape

def predict_mask(model, tensor_img, device, threshold=0.5):
    logger.info("Starte KI-Vorhersage...")
    tensor_img = tensor_img.to(device)
    with torch.no_grad():
        logits = model(tensor_img)
        probs = torch.sigmoid(logits)
        prediction_mask = (probs > threshold).squeeze().cpu().numpy()
    return prediction_mask

def apply_alignment(mask, original_h, original_w, x_shift, y_shift, scale):
    logger.info(f"Wende Affine Transformation an (Skalierung: {scale}, X: {x_shift}px, Y: {y_shift}px)...")
    center_y, center_x = original_h / 2.0, original_w / 2.0
    
    transform_matrix = np.array([
        [1.0 / scale, 0],
        [0, 1.0 / scale]
    ])
    
    offset_y = center_y - (center_y / scale) - y_shift
    offset_x = center_x - (center_x / scale) - x_shift
    
    aligned_mask = scipy.ndimage.affine_transform(
        mask,
        matrix=transform_matrix,
        offset=[offset_y, offset_x],
        order=0,
        mode='constant', 
        cval=0.0
    )
    return aligned_mask

def extract_rgb(img_t):
    r, g, b = img_t[:,:,2], img_t[:,:,1], img_t[:,:,0]
    rgb = np.dstack((r, g, b))
    rgb = (rgb / rgb.max() * 255).astype(np.uint8)
    return rgb

def visualize_inference(rgb, aligned_mask, x_shift, y_shift):
    logger.info("Generiere Overlay für Stakeholder...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1.imshow(rgb)
    ax1.set_title("Original Satellitenbild")
    ax1.axis("off")
    
    ax2.imshow(rgb)
    ax2.imshow(aligned_mask, cmap='Reds', alpha=0.5)
    ax2.set_title(f"KI-Detektion (Korrigiert um X:{x_shift}, Y:{y_shift})")
    ax2.axis("off")
    
    plt.tight_layout()
    plt.show()

def run_inference(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = config['paths']['model_weights']
    image_path = config['paths']['sentinel_image']
    norm_factor = config['data']['normalization_factor']
    threshold = config['inference']['threshold']
    
    # Interaktiver Versatz - standard Parameter für Kommandozeile
    x_shift, y_shift, scale = -8, -6, 1.04
    
    model = load_model(model_path, device)
    tensor_img, img_t, original_shape = load_and_preprocess_image(image_path, norm_factor)
    
    prediction_mask_padded = predict_mask(model, tensor_img, device, threshold)
    
    # Remove padding
    c, h, w = original_shape
    prediction_mask = prediction_mask_padded[:h, :w]
    
    aligned_mask = apply_alignment(prediction_mask, h, w, x_shift, y_shift, scale)
    
    rgb = extract_rgb(img_t)
    visualize_inference(rgb, aligned_mask, x_shift, y_shift)

if __name__ == "__main__":
    SRC_DIR = Path(__file__).resolve().parent
    ROOT_DIR = SRC_DIR.parent
    config_path = ROOT_DIR / 'config.yaml'
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
        
    config_data['paths']['model_weights'] = str(ROOT_DIR / config_data['paths']['model_weights'])
    config_data['paths']['sentinel_image'] = str(ROOT_DIR / config_data['paths']['sentinel_image'])
    
    run_inference(config_data)