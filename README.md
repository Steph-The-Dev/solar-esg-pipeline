# Solar-ESG-Pipeline: Cloud-Native Urban Deep Learning 🛰️

An end-to-end Machine Learning pipeline for automated building footprint extraction using multi-spectral Sentinel-2 satellite imagery and Deep Learning (U-Net).

Developed as a Proof of Concept (PoC) for the Basel urban perimeter, this project bridges the gap between raw optical physics, advanced geospatial engineering, and actionable ESG (Environmental, Social, and Governance) strategy.

## 🏗️ Modernized Architecture

The codebase has been refactored to meet enterprise standards, focusing on stability, validation, and reproducibility.

- **Strict Validation:** Integrated **Pydantic V2** for rigorous configuration validation and type safety.
- **Package-Centric:** Structured as a proper Python package for robust internal imports and installation.
- **Automated Quality:** Integrated **Ruff** for linting/formatting and **GitHub Actions** for CI/CD.
- **Lazy-Loading Pipeline:** Custom PyTorch DataLoaders with on-the-fly window reading to handle large GeoTIFFs without memory overflow.

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, Segmentation-Models-PyTorch, Albumentations.
* **Geospatial:** Rasterio, GeoPandas, OSMnx, Shapely.
* **Validation & Tooling:** Pydantic V2, Pytest, Ruff, YAML.
* **Infrastructure:** GitHub Actions, Mamba/Conda.

## 🚀 Quickstart

### 1. Rebuild Infrastructure
This project utilizes `mamba` for strict dependency management, specifically for handling complex geospatial C++ libraries (GDAL).

```bash
# Clone the repository
git clone https://github.com/Steph-The-Dev/solar-esg-pipeline.git
cd solar-esg-pipeline

# Create and activate the environment
mamba env create -f environment.local.yml
mamba activate solar-esg-env

# Install the project in editable mode
pip install -e .
```

### 2. Execute the Pipeline
The pipeline is managed via `config.yaml`. All paths and hyperparameters are validated at runtime.

```bash
# 1. Download Sentinel-2 imagery (COG API)
python src/download_sentinel.py

# 2. Generate aligned building masks from OSM
python src/align_and_mask.py

# 3. Train the U-Net model
python src/train.py

# 4. Run the Streamlit Dashboard
streamlit run src/app.py
```

## 🧪 Testing & Quality
Automated checks maintain a high standard of code quality.

```bash
# Run the test suite
pytest tests/

# Run the linter (Ruff)
ruff check src/
```

## 📂 Project Structure
```text
├── .github/workflows/   # CI/CD Pipeline
├── src/
│   ├── config_schema.py # Pydantic V2 Models
│   ├── dataset.py       # Lazy-loading PyTorch Dataset
│   ├── train.py         # Training Loop
│   ├── inference.py     # Post-processing & Alignment
│   └── app.py           # Streamlit Dashboard
├── tests/               # Pytest Suite (Mocked Data)
├── config.yaml          # Central Configuration
└── pyproject.toml       # Modern Packaging Metadata
```

---
*Built with a focus on clean-slate architecture, explainable AI, and cross-domain imaging scalability.*
