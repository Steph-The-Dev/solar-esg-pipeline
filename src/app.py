import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
import logging

from inference import (
    load_model as inf_load_model,
    load_and_preprocess_image as inf_load_and_preprocess,
    predict_mask,
    apply_alignment,
    extract_rgb
)

# --- KONFIGURATION ---
st.set_page_config(page_title="Solar ESG Pipeline - Basel PoC", layout="wide")

st.title("🛰️ Solar ESG Intelligence Dashboard")
st.markdown("""
    **Strategic Prototype:** Detektion von Gebäude-Footprints via Sentinel-2 & Deep Learning.
    Dieses Tool übersetzt multispektrale Satellitendaten in urbane Nachhaltigkeits-Metriken.
""")
st.markdown("---")

# --- CACHING (Modell & Daten) ---
@st.cache_resource
def load_config():
    with open('../config.yaml', 'r') as f:
        return yaml.safe_load(f)

config = load_config()

@st.cache_resource
def load_cached_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return inf_load_model(model_path, device), device

@st.cache_data
def load_cached_data(image_path, norm_factor):
    tensor_img, img_t, original_shape = inf_load_and_preprocess(image_path, norm_factor)
    return tensor_img, img_t, original_shape

# --- BUSINESS LOGIK ---
def calculate_esg_metrics(mask, pixel_area, irradiance, efficiency, co2_intensity):
    total_roof_area = np.sum(mask) * pixel_area
    annual_energy_mwh = (total_roof_area * irradiance * (efficiency / 100)) / 1000
    # CO2 Ersparnis in Tonnen
    co2_saved_tons = (annual_energy_mwh * co2_intensity) / 1000
    return total_roof_area, annual_energy_mwh, co2_saved_tons

# --- SIDEBAR: UI CONTROLS ---
st.sidebar.header("🔧 Visuelle Steuerung")
show_overlay = st.sidebar.toggle("KI-Overlay anzeigen", value=True)
alpha = st.sidebar.slider("Transparenz Maske", 0.0, 1.0, 0.4)
threshold = st.sidebar.slider("KI-Konfidenz (Threshold)", 0.1, 0.9, config['inference']['threshold'])

st.sidebar.header("📏 Geometrisches Alignment")
x_shift = st.sidebar.slider("X-Versatz (Pixel)", -10, 10, 3)
y_shift = st.sidebar.slider("Y-Versatz (Pixel)", -10, 10, -2)
scale = st.sidebar.slider("Metrische Skalierung", 0.95, 1.05, 1.05, 0.01)

st.sidebar.header("☀️ ESG & Solar-Parameter")
efficiency = st.sidebar.slider("Panel-Effizienz (%)", 10, 25, 18)
irradiance = st.sidebar.number_input("Einstrahlung (kWh/m²/Jahr)", value=1100, help="Durchschnittswert für das lokale Einzugsgebiet.")
co2_intensity = st.sidebar.number_input("CO₂-Intensität Strommix (kg/MWh)", value=128, help="Referenzwert: Schweizer Konsum-Mix liegt bei ca. 128 kg CO₂/MWh.")

# --- INFERENCE PIPELINE ---
model, device = load_cached_model(config['paths']['model_weights'])
tensor_img, img_t, original_shape = load_cached_data(config['paths']['sentinel_image'], config['data']['normalization_factor'])

# Vorhersage
prediction_mask_padded = predict_mask(model, tensor_img, device, threshold)
c, h, w = original_shape
prediction_mask = prediction_mask_padded[:h, :w]

# Alignment
aligned_mask = apply_alignment(prediction_mask, h, w, x_shift, y_shift, scale)
final_mask = (aligned_mask > threshold).astype(np.uint8)

# Metriken berechnen
pixel_area = 100 # 10m x 10m Auflösung bei Sentinel-2
total_roof_area, annual_energy_mwh, co2_saved_tons = calculate_esg_metrics(
    final_mask, pixel_area, irradiance, efficiency, co2_intensity
)

# --- LAYOUT & RENDER ---
col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(10, 10))
    rgb = extract_rgb(img_t)
    ax.imshow(rgb)
    if show_overlay:
        ax.imshow(final_mask, cmap='Reds', alpha=alpha)
    ax.axis('off')
    st.pyplot(fig, use_container_width=True)
    plt.close(fig) # WICHTIG: Vermeidet den Memory Leak!

with col2:
    st.subheader("Key Performance Indicators")
    st.metric("Detektierte Dachfläche", f"{total_roof_area:,.0f} m²")
    st.metric("Solar-Potenzial (Ertrag)", f"{annual_energy_mwh:,.1f} MWh/Jahr")
    st.metric("CO₂-Ersparnis (Äquivalent)", f"{co2_saved_tons:,.1f} Tonnen/Jahr")
    
    st.info("""
    **Architektur & Setup:**
    - Sensorik: ESA Sentinel-2 (Multispektral)
    - Architektur: U-Net++ (ResNet34 Backbone)
    - Framework: PyTorch mit hardware-beschleunigter Inference
    """)
    
    # CSV Export
    export_data = f"Area_m2,Energy_MWh,CO2_Saved_Tons\n{total_roof_area},{annual_energy_mwh},{co2_saved_tons}"
    st.download_button("KPIs exportieren (CSV)", export_data, "esg_metrics_export.csv")