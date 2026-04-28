import logging
import sys
from pathlib import Path

# Fix for Streamlit Cloud: Add project root to sys.path
# This allows 'from src.config_schema import ...' to work even if the app is run from inside 'src/'
# or if 'src/' is the working directory.
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from src.config_schema import get_resolved_config
from src.inference import (
    apply_alignment,
    extract_rgb,
    predict_mask,
)
from src.inference import load_and_preprocess_image as inf_load_and_preprocess
from src.inference import load_model as inf_load_model

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def load_cached_model(model_path, device):
    return inf_load_model(model_path, device), device


@st.cache_data
def load_cached_data(image_path, norm_factor):
    return inf_load_and_preprocess(image_path, norm_factor)


# --- UI LAYOUT ---
st.set_page_config(page_title="Solar-ESG Pipeline 🛰️", layout="wide")

st.title("Solar-ESG: AI-based Building Detection")
st.markdown(
    """
This app uses a **Deep Learning model (U-Net)** to extract building footprints from
Sentinel-2 satellite data. The results serve as a basis for ESG analyses
(photovoltaic potential, CO2 savings).
"""
)

# Resolve paths
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
config_path = ROOT_DIR / "config.yaml"
config = get_resolved_config(str(config_path), ROOT_DIR)

model_path = config.paths.model_weights
image_path = config.paths.sentinel_image

# --- SIDEBAR ---
st.sidebar.header("⚙️ Parameters")
show_overlay = st.sidebar.toggle("Show AI Overlay", value=True)
alpha = st.sidebar.slider("Mask Transparency", 0.0, 1.0, 0.4)
threshold = st.sidebar.slider(
    "AI Confidence (Threshold)", 0.1, 0.9, config.inference.threshold
)

st.sidebar.header("📏 Geometric Alignment")
x_shift = st.sidebar.slider("X-Offset (Pixels)", -50.0, 50.0, value=float(config.inference.x_shift))
y_shift = st.sidebar.slider("Y-Offset (Pixels)", -50.0, 50.0, value=float(config.inference.y_shift))
scale = st.sidebar.slider("Scaling Factor", 0.9, 1.1, config.inference.scale, step=0.01)

st.sidebar.header("🌱 ESG Metrics")
avg_yield = st.sidebar.slider("Spec. Yield (kWh/m²/year)", 500, 2000, value=1100)
co2_factor = st.sidebar.slider("CO2 Intensity Mix (kg/MWh)", 0, 500, value=128)

# --- PIPELINE EXECUTION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with st.spinner("Loading model and data..."):
    model, device = load_cached_model(model_path, device)
    tensor_img, img_t, original_shape = load_cached_data(
        image_path, config.data.normalization_factor
    )

# Prediction
with st.spinner("AI analyzing image..."):
    prediction_mask_padded = predict_mask(model, tensor_img, device, threshold)
    _, h, w = original_shape
    prediction_mask = prediction_mask_padded[:h, :w]
    aligned_mask = apply_alignment(prediction_mask, h, w, x_shift, y_shift, scale)

# Visualization
col1, col2 = st.columns(2)

rgb = extract_rgb(img_t)

with col1:
    st.subheader("Satellite Image (RGB)")
    st.image(rgb, width='stretch')

with col2:
    st.subheader("AI Detection & ESG Potential")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb)
    if show_overlay:
        # Visualize mask (buildings in red)
        masked = np.ma.masked_where(aligned_mask == 0, aligned_mask)
        ax.imshow(masked, cmap="Reds", alpha=alpha)
    ax.axis("off")
    st.pyplot(fig)

# --- ESG DASHBOARD ---
st.divider()
st.subheader("📊 ESG Insights (Basel Demo)")

total_roof_area = np.sum(aligned_mask) * 100  # Rough estimate: 10m x 10m per pixel
annual_energy_mwh = (total_roof_area * avg_yield) / 1000000
co2_saved_tons = (annual_energy_mwh * co2_factor) / 1000

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Detected Roof Area", f"{total_roof_area:,.0f} m²")
kpi2.metric("Potential Energy", f"{annual_energy_mwh:,.1f} MWh/year")
kpi3.metric("CO2 Saved", f"{co2_saved_tons:,.1f} Tons/year")

with st.expander("Details on Calculation"):
    num_pixels = np.sum(aligned_mask)
    st.write(
        f"""
    - **Roof Area:** Based on {num_pixels} detected pixels à 100m² (Sentinel-2 resolution).
    - **Energy:** Roof area * {avg_yield} kWh/m² yield.
    - **CO2:** Energy * {co2_factor} kg CO2 saving per MWh (Swiss Mix).
    """
    )
    # CSV Export
    export_csv = (
        f"Area_m2,Energy_MWh,CO2_Saved_Tons\n"
        f"{total_roof_area},{annual_energy_mwh},{co2_saved_tons}"
    )
    st.download_button("Export KPIs (CSV)", export_csv, "esg_metrics_export.csv")
