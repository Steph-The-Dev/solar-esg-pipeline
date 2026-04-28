import logging
from pathlib import Path

import albumentations as alb
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2

import rasterio
from src.config_schema import FullConfig, get_resolved_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    """
    Loads the U-Net model and its weights.

    Args:
        weights_path: Path to the .pth weights file.
        device: CPU or GPU device.

    Returns:
        The model in evaluation mode.
    """
    logger.info("Loading model and weights...")
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=4, classes=1).to(
        device
    )
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    return model


def load_and_preprocess_image(
    image_path: str, normalization_factor: float
) -> tuple[torch.Tensor, np.ndarray, tuple]:
    """
    Loads a GeoTIFF and prepares it for model inference.

    Args:
        image_path: Path to the satellite image.
        normalization_factor: Factor for radiometric calibration.

    Returns:
        A tuple of (input_tensor, visualization_array, original_shape).
    """
    logger.info("Loading test image...")
    with rasterio.open(image_path) as src:
        img_data = src.read()

    original_shape = img_data.shape
    img_data = img_data.astype(np.float32) / normalization_factor
    img_data = np.clip(img_data, 0.0, 1.0)
    img_t = np.transpose(img_data, (1, 2, 0))

    transform = alb.Compose(
        [alb.PadIfNeeded(min_height=128, min_width=128, border_mode=0, fill=0), ToTensorV2()]
    )
    tensor_img = transform(image=img_t)["image"].unsqueeze(0)
    return tensor_img, img_t, original_shape


def predict_mask(
    model: torch.nn.Module, tensor_img: torch.Tensor, device: torch.device, threshold: float = 0.5
) -> np.ndarray:
    """
    Performs the AI prediction.

    Args:
        model: Trained PyTorch model.
        tensor_img: Preprocessed input tensor.
        device: Computing device.
        threshold: Confidence threshold for binary mask.

    Returns:
        Predicted binary mask as a NumPy array.
    """
    logger.info("Starting AI prediction...")
    tensor_img = tensor_img.to(device)
    with torch.no_grad():
        logits = model(tensor_img)
        probs = torch.sigmoid(logits)
        prediction_mask = (probs > threshold).squeeze().cpu().numpy()
    return prediction_mask


def apply_alignment(
    mask: np.ndarray, original_h: int, original_w: int, x_shift: float, y_shift: float, scale: float
) -> np.ndarray:
    """
    Applies an affine transformation to correct for satellite parallax shifts.

    Args:
        mask: Predicted binary mask.
        original_h, original_w: Original dimensions.
        x_shift, y_shift: Translation in pixels.
        scale: Scaling factor.

    Returns:
        Aligned binary mask.
    """
    logger.info(
        f"Applying affine transformation (Scale: {scale}, X: {x_shift}px, Y: {y_shift}px)..."
    )
    center_y, center_x = original_h / 2.0, original_w / 2.0

    transform_matrix = np.array([[1.0 / scale, 0], [0, 1.0 / scale]])

    offset_y = center_y - (center_y / scale) - y_shift
    offset_x = center_x - (center_x / scale) - x_shift

    aligned_mask = scipy.ndimage.affine_transform(
        mask,
        matrix=transform_matrix,
        offset=[offset_y, offset_x],
        order=0,
        mode="constant",
        cval=0.0,
    )
    return aligned_mask


def extract_rgb(img_t: np.ndarray) -> np.ndarray:
    """
    Extracts RGB bands for visualization.
    """
    r, g, b = img_t[:, :, 2], img_t[:, :, 1], img_t[:, :, 0]
    rgb = np.dstack((r, g, b))
    rgb = (rgb / (rgb.max() if rgb.max() > 0 else 1.0) * 255).astype(np.uint8)
    return rgb


def visualize_inference(
    rgb: np.ndarray, aligned_mask: np.ndarray, x_shift: float, y_shift: float
) -> None:
    """
    Generates a visual comparison between the original image and the detection.
    """
    logger.info("Generating overlay for stakeholders...")
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    ax1.imshow(rgb)
    ax1.set_title("Original Satellite Image")
    ax1.axis("off")

    ax2.imshow(rgb)
    ax2.imshow(aligned_mask, cmap="Reds", alpha=0.5)
    ax2.set_title(f"AI Detection (Corrected X:{x_shift}, Y:{y_shift})")
    ax2.axis("off")

    plt.tight_layout()
    plt.show()


def run_inference(config: FullConfig) -> None:
    """
    Main entry point for the inference pipeline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = config.paths.model_weights
    image_path = config.paths.sentinel_image
    norm_factor = config.data.normalization_factor
    threshold = config.inference.threshold

    # Use offsets from configuration
    x_shift = config.inference.x_shift
    y_shift = config.inference.y_shift
    scale = config.inference.scale

    model = load_model(model_path, device)
    tensor_img, img_t, original_shape = load_and_preprocess_image(image_path, norm_factor)

    prediction_mask_padded = predict_mask(model, tensor_img, device, threshold)

    # Remove padding
    _, h, w = original_shape
    prediction_mask = prediction_mask_padded[:h, :w]

    aligned_mask = apply_alignment(prediction_mask, h, w, x_shift, y_shift, scale)

    rgb = extract_rgb(img_t)
    visualize_inference(rgb, aligned_mask, x_shift, y_shift)


if __name__ == "__main__":
    SRC_DIR = Path(__file__).resolve().parent
    ROOT_DIR = SRC_DIR.parent
    config_path = ROOT_DIR / "config.yaml"

    config_data: FullConfig = get_resolved_config(str(config_path), ROOT_DIR)

    run_inference(config_data)
