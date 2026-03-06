# SRIP 2026 — AI for Sustainability
## Earth Observation: Delhi Airshed Land Use Classification

**Applicant:** Pranati Sharma
**Institution:** Guru Gobind Singh Indraprastha University
**Program:** B.Tech in Industrial Internet of Things (IIoT)
**Project:** IP0NB0000021 — AI for Sustainability (Prof. Nipun Batra, IIT Gandhinagar)

---

## AI Tool Disclosure
As permitted by the assignment guidelines, this code was developed with assistance from Claude (Anthropic AI). All code has been reviewed, understood, and verified by the author. The author takes full responsibility for the implementation and is prepared to explain every component during the one-on-one discussion.

---

## Problem Statement
Build a full pipeline for spatial gridding and land cover classification of the Delhi Airshed region using Sentinel-2 satellite imagery and ESA WorldCover 2021 data.

---

## Dataset
| File | Description |
|------|-------------|
| `delhi_ncr_region.geojson` | Shapefile of Delhi-NCR region (EPSG:4326) |
| `delhi_airshed.geojson` | Shapefile of Delhi Airshed region (EPSG:4326) |
| `worldcover_bbox_delhi_ncr_2021.tif` | ESA WorldCover 2021 land cover raster (10m resolution) |
| `rgb/` | Sentinel-2 RGB image patches (128×128 pixels, 10m/pixel) |

Source: [Kaggle — Earth Observation Delhi Airshed](https://www.kaggle.com/datasets/rishabhsnip/earth-observation-delhi-airshed)

---

## Pipeline Overview

### Q1 — Spatial Reasoning & Data Filtering
- Plots Delhi-NCR shapefile with a 60×60 km uniform grid overlay
- Filters satellite images whose center coordinates fall inside the region
- Reports image count before and after filtering

### Q2 — Label Construction & Dataset Preparation
- Extracts 128×128 land cover patches from `land_cover.tif` for each image
- Assigns labels using dominant (mode) ESA class per patch
- Maps ESA codes to 5 simplified categories: Built-up, Vegetation, Cropland, Water, Others
- Performs 60/40 train-test split with class distribution visualization

### Q3 — Model Training & Supervised Evaluation
- Trains ResNet18 (pretrained, fine-tuned) for land use classification
- Evaluates using Accuracy and Weighted F1-Score
- Displays and interprets confusion matrix

---

## ESA WorldCover Class Mapping
| ESA Code | Category |
|----------|----------|
| 10 | Vegetation |
| 20 | Vegetation |
| 30 | Vegetation |
| 40 | Cropland |
| 50 | Built-up |
| 60 | Others |
| 70 | Others |
| 80 | Water |
| 90 | Others |
| 95 | Others |
| 100 | Others |

---

## Requirements
```bash
pip install geopandas rasterio shapely matplotlib numpy pandas scikit-learn torch torchvision pillow scipy pyproj
```

---

## How to Run
```bash
python srip_assignment.py
```

### Output Files
| File | Description |
|------|-------------|
| `q1_delhi_ncr_grid.png` | Delhi-NCR map with 60×60 km grid |
| `q2_class_distribution.png` | Train/test class distribution plots |
| `q3_training_loss.png` | Training loss curve |
| `q3_confusion_matrix.png` | Confusion matrix |
| `land_cover_resnet18.pth` | Trained model weights |

---

## Model Architecture
- **Base:** ResNet18 (ImageNet pretrained)
- **Final layer:** Linear(512 → 5 classes)
- **Optimizer:** Adam (lr=1e-4)
- **Scheduler:** StepLR (step=5, gamma=0.5)
- **Epochs:** 10
- **Batch size:** 16
