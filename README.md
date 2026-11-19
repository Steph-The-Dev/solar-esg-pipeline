# Solar-ESG Pipeline
 
A proof of concept for building footprint segmentation in Basel from Sentinel-2 satellite imagery. A U-Net is trained on OpenStreetMap building outlines, and a Streamlit dashboard turns the predicted roof area into rough estimates of solar potential and CO₂ savings.
 
The goal was to build a complete pipeline end to end — from downloading the data to an interactive result — and to learn how imaging questions such as band selection, radiometric scaling and image registration play out in remote sensing.
 
<!-- Add a screenshot of the dashboard here, e.g. docs/dashboard.png -->
 
## Pipeline
 
```text
STAC search (Sentinel-2 L2A) → 4-band crop (B, G, R, NIR) → OSM building masks → U-Net training → inference + alignment → dashboard
```
 
1. **Data download** (`download_sentinel.py`): Searches the Element 84 Earth Search STAC API for the Sentinel-2 L2A scene with the lowest cloud cover in the configured period (default: summer 2023, < 20 % cloud cover). The blue, green, red and near-infrared bands are read directly from Cloud-Optimized GeoTIFFs and cropped to the Basel bounding box.
2. **Labels** (`align_and_mask.py`): Building footprints from OpenStreetMap (via OSMnx), reprojected to the image CRS and rasterized into a binary mask.
3. **Dataset** (`dataset.py`): Random 64×64 patches read lazily from disk with rasterio windows, so large GeoTIFFs don't have to fit in memory. Reflectance values are scaled by a fixed factor and clipped to [0, 1]. Augmentation: flips and 90° rotations.
4. **Training** (`train.py`): U-Net with a ResNet-34 encoder (segmentation-models-pytorch), 4 input channels, trained from scratch with Dice loss.
5. **Inference** (`inference.py`): Sigmoid output and threshold, followed by an affine correction (shift and scale) to align the predicted mask with the image.
6. **Dashboard** (`app.py`): Overlay of the prediction on the RGB image with adjustable threshold, alignment and transparency, plus estimates for roof area, annual energy yield and CO₂ savings, exportable as CSV.
All paths and hyperparameters live in `config.yaml` and are validated at startup with Pydantic.
 
## Tech stack
 
- **Deep learning:** PyTorch, segmentation-models-pytorch, Albumentations
- **Geospatial:** rasterio, GeoPandas, OSMnx, Shapely, STAC API
- **Tooling:** Pydantic, pytest, ruff, GitHub Actions, conda/mamba
## Getting started
 
```bash
git clone https://github.com/Steph-The-Dev/solar-esg-pipeline.git
cd solar-esg-pipeline
 
mamba env create -f environment.local.yml
mamba activate solar-esg-env
pip install -e .
```
 
The conda environment is used for local development because GDAL-based libraries are easier to install that way. `requirements.txt` is a lighter CPU-only setup for deployment.
 
```bash
python src/download_sentinel.py   # 1. download Sentinel-2 bands
python src/align_and_mask.py      # 2. create building masks from OSM
python src/train.py               # 3. train the U-Net
streamlit run src/app.py          # 4. start the dashboard
```
 
A pretrained proof-of-concept model and the cropped Sentinel-2 image are included, so the dashboard runs without retraining.
 
## Testing
 
```bash
pytest tests/
ruff check src/
```
 
The tests cover config validation, model architecture and the dataset logic, using synthetic GeoTIFFs. CI runs linting and tests on every push.
 
## Limitations
 
This is a proof of concept, and the numbers should be read that way.
 
- **No quantitative evaluation yet.** There is no validation split and no IoU or F1 score — only the training loss is logged. How well the model segments buildings is therefore not measured.
- **Small training setup.** One scene, one city, 10 epochs of 200 random patches, no pretrained encoder weights. Training and inspection happen on the same area.
- **Resolution.** Sentinel-2 has 10 m pixels, so many individual buildings span only a few pixels. The result is a coarse built-up mask, not precise roof outlines.
- **Registration is corrected by hand.** OSM outlines and the satellite image don't line up exactly. The offset is currently compensated after prediction with a manually tuned affine shift (x −8 px, y −6 px, scale 1.04). The cleaner solution is to align the labels before training.
- **The energy figures are an order of magnitude, not an assessment.** They assume every detected pixel is usable roof (100 m² each) with a single specific yield and CO₂ factor. Roof orientation, pitch, shading and usable area are not considered.
## Next steps
 
- Hold out a separate tile for validation and report IoU and F1.
- Align OSM labels with the imagery before training instead of correcting predictions afterwards.
- Test whether a pretrained encoder and the additional Sentinel-2 bands (red edge, SWIR) improve the segmentation.
 
