import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import scipy.ndimage # NEU: Für die Pixel-Verschiebung

def run_inference():
    print("🧠 Lade Modell und Gewichte...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Architektur exakt wie im Training initialisieren
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=4,
        classes=1
    ).to(device)
    
    # Trainierte Gewichte laden
    model.load_state_dict(torch.load("../models/solar_unet_poc.pth", weights_only=True))
    model.eval() # WICHTIG: Schaltet Dropout und BatchNorm ab für deterministische Vorhersagen
    
    print("🌍 Lade Testbild (Basel Sentinel-2)...")
    with rasterio.open("../data/raw/basel_sentinel2_cropped.tif") as src:
        img_data = src.read()
        
    original_c, original_h, original_w = img_data.shape
        
    # 2. Radiometrische Normalisierung (Muss exakt dem Training entsprechen!)
    img_data = img_data.astype(np.float32) / 4000.0
    img_data = np.clip(img_data, 0.0, 1.0)
    img_transposed = np.transpose(img_data, (1, 2, 0)) # Zu (H, W, C)
    
    # 3. Der Pad & Convert Trick für das U-Net
    transform = A.Compose([
        A.PadIfNeeded(min_height=128, min_width=128, border_mode=0, fill=0), 
        ToTensorV2()
    ])
    
    tensor_img = transform(image=img_transposed)['image'].unsqueeze(0).to(device)
    
    print("⚡ Starte KI-Vorhersage auf der RTX 3080 Ti...")
    with torch.no_grad(): # Blockiert die Gradientenberechnung -> Spart VRAM und ist massiv schneller
        logits = model(tensor_img)
        # Logits durch Sigmoid jagen -> Gibt uns Wahrscheinlichkeiten zwischen 0.0 und 1.0
        probs = torch.sigmoid(logits)
        # Binarisierung: Alles über 50% Wahrscheinlichkeit ist ein Dach
        prediction_mask = (probs > 0.5).squeeze().cpu().numpy()
        
    # Wir schneiden den schwarzen Padding-Rand wieder weg, um die Originalgröße herzustellen
    prediction_mask = prediction_mask[:original_h, :original_w]
    
    print("🎨 Generiere Overlay für Stakeholder...")
    # RGB-Bild für den visuellen Hintergrund extrahieren und normieren
    r, g, b = img_transposed[:,:,2], img_transposed[:,:,1], img_transposed[:,:,0]
    rgb = np.dstack((r, g, b))
    rgb = (rgb / rgb.max() * 255).astype(np.uint8)
    
    # --- DER MANUELLE ALIGNMENT FIX (V2: Affine Transformation) ---
    import scipy.ndimage # Falls du es noch nicht ganz oben importiert hast
    
    # 1. Deine Parameter (Hier kannst du iterieren)
    X_VERSATZ = -7        # Positiv = rechts, Negativ = links
    Y_VERSATZ = -6      # Positiv = unten, Negativ = oben
    SKALIERUNG = 1.03   # 1.05 = 5% größer, 0.95 = 5% kleiner
    
    print(f"🔧 Wende Affine Transformation an (Skalierung: {SKALIERUNG}, X: {X_VERSATZ}px, Y: {Y_VERSATZ}px)...")
    
    # 2. Zentrum des Bildes berechnen (Wir wollen aus der Mitte heraus skalieren, nicht aus der Ecke)
    center_y, center_x = original_h / 2.0, original_w / 2.0
    
    # 3. Die Transformations-Matrix aufbauen
    # scipy arbeitet "rückwärts" (Mapping von Output zu Input), daher teilen wir durch die Skalierung
    transform_matrix = np.array([
        [1.0 / SKALIERUNG, 0],
        [0, 1.0 / SKALIERUNG]
    ])
    
    # 4. Den Offset berechnen (Kombination aus Skalierungs-Zentrierung und deinem manuellen Shift)
    offset_y = center_y - (center_y / SKALIERUNG) - Y_VERSATZ
    offset_x = center_x - (center_x / SKALIERUNG) - X_VERSATZ
    
    # 5. Transformation ausführen
    aligned_mask = scipy.ndimage.affine_transform(
        prediction_mask,
        matrix=transform_matrix,
        offset=[offset_y, offset_x],
        order=0,            # Order=0 ist extrem wichtig: Es verhindert verschwommene, graue Ränder bei harten Masken
        mode='constant', 
        cval=0.0            # Alles, was "neu" ins Bild geschoben wird, wird schwarz (0.0) gefüllt
    )
    # ----------------------------------------------------------------
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1.imshow(rgb)
    ax1.set_title("Original Satellitenbild")
    ax1.axis("off")
    
    ax2.imshow(rgb)
    # WICHTIG: Wir plotten jetzt die verschobene (aligned) Maske!
    ax2.imshow(aligned_mask, cmap='Reds', alpha=0.5)
    ax2.set_title(f"KI-Detektion (Korrigiert um X:{X_VERSATZ}, Y:{Y_VERSATZ})")
    ax2.axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_inference()