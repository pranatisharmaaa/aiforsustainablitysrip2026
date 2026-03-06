# ============================================================
# SRIP 2026 - AI for Sustainability
# Earth Observation Assignment
# Author: Pranati Sharma
# Note: Code written with assistance from Claude (Anthropic AI).
#       All code has been reviewed and understood by the author.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from scipy import stats
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from pyproj import Transformer

# ============================================================
# CONFIGURATION - Update these paths as needed
# ============================================================
DELHI_NCR_SHP     = "delhi_ncr_region.geojson"
AIRSHED_SHP       = "delhi_airshed.geojson"
LAND_COVER_TIF    = "worldcover_bbox_delhi_ncr_2021.tif"
RGB_DIR           = "rgb"
COORDS_CSV        = "rgb/coordinates.csv"   # CSV with columns: filename, latitude, longitude
RANDOM_SEED       = 42
BATCH_SIZE        = 16
EPOCHS            = 10
LEARNING_RATE     = 1e-4
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ESA WorldCover class mapping
ESA_CLASS_MAP = {
    10: "Vegetation",
    20: "Vegetation",
    30: "Vegetation",
    40: "Cropland",
    50: "Built-up",
    60: "Others",
    70: "Others",
    80: "Water",
    90: "Others",
    95: "Others",
    100: "Others",
}
LABEL_NAMES = ["Built-up", "Cropland", "Vegetation", "Water", "Others"]
LABEL_TO_IDX = {name: idx for idx, name in enumerate(LABEL_NAMES)}


# ============================================================
# Q1: SPATIAL REASONING & DATA FILTERING
# ============================================================

def q1_spatial_filtering():
    print("\n======== Q1: Spatial Reasoning & Data Filtering ========")

    # Load shapefiles
    delhi_ncr = gpd.read_file(DELHI_NCR_SHP).to_crs(epsg=4326)
    airshed   = gpd.read_file(AIRSHED_SHP).to_crs(epsg=4326)

    # --- Plot Delhi-NCR with 60x60 km grid ---
    # Reproject to metric CRS for gridding
    delhi_metric = delhi_ncr.to_crs(epsg=32644)
    bounds = delhi_metric.total_bounds  # minx, miny, maxx, maxy
    grid_size = 60_000  # 60 km in meters

    # Generate grid lines
    x_lines = np.arange(bounds[0], bounds[2] + grid_size, grid_size)
    y_lines = np.arange(bounds[1], bounds[3] + grid_size, grid_size)

    # Convert grid lines back to WGS84 for plotting
    transformer = Transformer.from_crs("EPSG:32644", "EPSG:4326", always_xy=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    delhi_ncr.plot(ax=ax, color="lightblue", edgecolor="black", linewidth=1.5, label="Delhi-NCR")
    airshed.plot(ax=ax, color="none", edgecolor="red", linewidth=1.5, linestyle="--", label="Delhi Airshed")

    # Draw grid lines in WGS84
    ncr_metric_bounds = delhi_metric.total_bounds
    for x in x_lines:
        lons, lats = transformer.transform(
            [x, x],
            [ncr_metric_bounds[1] - grid_size, ncr_metric_bounds[3] + grid_size]
        )
        ax.plot(lons, lats, color="gray", linewidth=0.5, linestyle="--", alpha=0.7)
    for y in y_lines:
        lons, lats = transformer.transform(
            [ncr_metric_bounds[0] - grid_size, ncr_metric_bounds[2] + grid_size],
            [y, y]
        )
        ax.plot(lons, lats, color="gray", linewidth=0.5, linestyle="--", alpha=0.7)

    ax.set_title("Delhi-NCR Region with 60×60 km Grid", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    plt.tight_layout()
    plt.savefig("q1_delhi_ncr_grid.png", dpi=150)
    plt.show()
    print("Saved: q1_delhi_ncr_grid.png")

    # --- Filter satellite images by coordinates ---
    coords_df = pd.read_csv(COORDS_CSV)
    print(f"Total images before filtering: {len(coords_df)}")

    # Convert coords to GeoDataFrame
    from shapely.geometry import Point
    geometry = [Point(lon, lat) for lat, lon in zip(coords_df["latitude"], coords_df["longitude"])]
    gdf_images = gpd.GeoDataFrame(coords_df, geometry=geometry, crs="EPSG:4326")

    # Filter: keep only images whose center falls inside Delhi-NCR
    delhi_union = delhi_ncr.union_all() if hasattr(delhi_ncr, 'union_all') else delhi_ncr.unary_union
    mask = gdf_images.geometry.within(delhi_union)
    filtered_df = gdf_images[mask].reset_index(drop=True)

    print(f"Total images after filtering: {len(filtered_df)}")
    return filtered_df


# ============================================================
# Q2: LABEL CONSTRUCTION & DATASET PREPARATION
# ============================================================

def extract_label(lat, lon, src):
    """Extract dominant land cover class for a 128x128 patch centered on (lat, lon)."""
    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    row, col = rowcol(src.transform, x, y)

    half = 64
    row_start = max(row - half, 0)
    row_end   = min(row + half, src.height)
    col_start = max(col - half, 0)
    col_end   = min(col + half, src.width)

    patch = src.read(1, window=rasterio.windows.Window(
        col_start, row_start,
        col_end - col_start, row_end - row_start
    ))

    if patch.size == 0:
        return None

    dominant_class = int(stats.mode(patch.flatten(), keepdims=True)[0][0])
    return dominant_class


def map_esa_to_category(esa_code):
    return ESA_CLASS_MAP.get(esa_code, "Others")


def q2_label_construction(filtered_df):
    print("\n======== Q2: Label Construction & Dataset Preparation ========")

    labels_esa  = []
    labels_cat  = []
    valid_files = []

    with rasterio.open(LAND_COVER_TIF) as src:
        for _, row in filtered_df.iterrows():
            dominant = extract_label(row["latitude"], row["longitude"], src)
            if dominant is not None:
                category = map_esa_to_category(dominant)
                labels_esa.append(dominant)
                labels_cat.append(category)
                valid_files.append(row["filename"])

    dataset_df = pd.DataFrame({
        "filename": valid_files,
        "latitude": filtered_df.loc[filtered_df["filename"].isin(valid_files), "latitude"].values,
        "longitude": filtered_df.loc[filtered_df["filename"].isin(valid_files), "longitude"].values,
        "esa_code": labels_esa,
        "category": labels_cat,
        "label_idx": [LABEL_TO_IDX[c] for c in labels_cat]
    })

    print(f"Total labeled images: {len(dataset_df)}")
    print("\nClass distribution:")
    print(dataset_df["category"].value_counts())

    # 60/40 train-test split
    train_df, test_df = train_test_split(
        dataset_df, test_size=0.4, random_state=RANDOM_SEED, stratify=dataset_df["label_idx"]
    )
    print(f"\nTrain size: {len(train_df)}, Test size: {len(test_df)}")

    # Visualize class distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    train_df["category"].value_counts().plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="black")
    axes[0].set_title("Train Set Class Distribution")
    axes[0].set_xlabel("Land Use Category")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis='x', rotation=45)

    test_df["category"].value_counts().plot(kind="bar", ax=axes[1], color="coral", edgecolor="black")
    axes[1].set_title("Test Set Class Distribution")
    axes[1].set_xlabel("Land Use Category")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig("q2_class_distribution.png", dpi=150)
    plt.show()
    print("Saved: q2_class_distribution.png")

    return train_df, test_df


# ============================================================
# Q3: MODEL TRAINING & SUPERVISED EVALUATION
# ============================================================

class SentinelDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        image    = Image.open(img_path).convert("RGB")
        label    = int(row["label_idx"])
        if self.transform:
            image = self.transform(image)
        return image, label


def build_model(num_classes):
    """Use pretrained ResNet18, replace final layer."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_model(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate_model(model, test_loader):
    model.eval()
    all_preds  = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def q3_model_training(train_df, test_df):
    print("\n======== Q3: Model Training & Supervised Evaluation ========")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SentinelDataset(train_df, RGB_DIR, transform=transform)
    test_dataset  = SentinelDataset(test_df,  RGB_DIR, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(LABEL_NAMES)
    model       = build_model(num_classes).to(DEVICE)
    criterion   = nn.CrossEntropyLoss()
    optimizer   = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler   = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print(f"Training on: {DEVICE}")
    train_losses = []
    for epoch in range(1, EPOCHS + 1):
        loss = train_model(model, train_loader, criterion, optimizer)
        scheduler.step()
        train_losses.append(loss)
        print(f"Epoch [{epoch}/{EPOCHS}] Loss: {loss:.4f}")

    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, EPOCHS + 1), train_losses, marker="o", color="steelblue")
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig("q3_training_loss.png", dpi=150)
    plt.show()

    # Evaluate
    preds, labels = evaluate_model(model, test_loader)
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="weighted", zero_division=0)

    print(f"\nTest Accuracy : {acc:.4f}")
    print(f"Weighted F1   : {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - Land Use Classification")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("q3_confusion_matrix.png", dpi=150)
    plt.show()
    print("Saved: q3_confusion_matrix.png")

    # Interpretation
    print("\n--- Results Interpretation ---")
    print(f"The model achieved {acc*100:.1f}% accuracy on the test set.")
    print("Classes with high recall (diagonal values) indicate the model learned those")
    print("land-use patterns well. Off-diagonal values indicate misclassifications,")
    print("often between visually similar classes such as Cropland and Vegetation.")

    # Save model
    torch.save(model.state_dict(), "land_cover_resnet18.pth")
    print("\nModel saved: land_cover_resnet18.pth")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Q1
    filtered_df = q1_spatial_filtering()

    # Q2
    train_df, test_df = q2_label_construction(filtered_df)

    # Q3
    q3_model_training(train_df, test_df)

    print("\n✅ All questions completed. Check output images and model file.")
