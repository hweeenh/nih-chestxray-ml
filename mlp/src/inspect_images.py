"""
# Script Overview
This script inspects actual x-ray images before building the MLP

## Features
- Obtains important properties of raw images
-- Dimensions: 1024 x 1024 high-resolution square images
-- Mode: L (Luminance, a single grayscale intensity channel) 8-bit grayscale
-- Format: PNG (Lossless image format)
-- Pixel range: 0-255 (256 possible intensity values)
"""

from pathlib import Path
from PIL import Image

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
IMAGE_DIR = DATA_DIR / "images"

# Inspect a few images
image_paths = sorted(IMAGE_DIR.glob("*png"))[:5]

for path in image_paths:
    with Image.open(path) as img:
        print(f"Image: {path.name}")
        print("Dimensions:", img.size)
        print("Image mode:", img.mode)
        print("Format:", img.format)

# Inspect one image's pixel values
sample_path = image_paths[0]

with Image.open(sample_path) as img:
    img = img.convert("L")
    print("Pixel range:", img.getextrema())
