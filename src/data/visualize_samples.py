"""
# Script Overview
This script shows the actual images/x-rays that the model will be learning from

## Features
- Displays 1 sample images from each class (Effusion and no findings)
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "raw"
IMAGE_DIR = DATA_DIR / "images"

df = pd.read_csv(PROJECT_DIR / "data" / "processed" / "mlp_dataset.csv")

# Select 1 image from each class
effusion = df[df["Label"] == 1].iloc[0]
no_finding = df[df["Label"] == 0].iloc[0]

samples = [
    ("No Finding", no_finding),
    ("Effusion", effusion),]

fig, axes = plt.subplots(1, 2, figsize=(10,5))

for ax, (title, row) in zip(axes, samples):
    image_path = IMAGE_DIR / row["Image Index"]
    with Image.open(image_path) as img:
        ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()        