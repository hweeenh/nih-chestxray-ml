"""
# Script Overview
This script preprocesses the x-ray images from 1024x1024 to 64x64 pixels.

## Features
- Resize: Reduce image dimensions to 64x64
- Convert to tensor: Represent image as a PyTorch tensor
- Normalize: Convert pixel values from 0-255 to 0-1
"""

from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

# 1. Paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_DIR / "data" / "raw" / "images"
CSV_PATH = PROJECT_DIR / "data" / "processed" / "mlp_dataset.csv"

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load dataset metadata
df = pd.read_csv(CSV_PATH)

# 3. Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),])

# 4. Process images
images = []
labels = []
for _, row in df.iterrows():
    image_path = IMAGE_DIR / row["Image Index"]
    with Image.open(image_path) as img:
        img = img.convert("L")
        image_tensor = transform(img)
    images.append(image_tensor)
    labels.append(row["Label"])

# 5. Convert lists into tensors
X = torch.stack(images)
y = torch.tensor(labels, dtype=torch.long)

# 6. Save processed tensors
torch.save({
    "X": X,
    "y": y,
    "image_names": df["Image Index"].tolist(),
    "patient_ids": df["Patient ID"].tolist(),
    "splits": df["Split"].tolist(),},
    OUTPUT_DIR / "mlp_images_64x64.pt")

# 7. Inspect results
print("Image tensor shape:", X.shape)   # 3241 grayscale images
print("Label tensor shape:", y.shape)   # 1 label per image
print("Pixel minimum:", X.min().item()) # 0 black
print("Pixel maximum:", X.max().item()) # 1 white (0-255 normalised to 0-1)
print(f"Label counts: {y.bincount()}")  # 2754 No Findings, 487 Effusions
