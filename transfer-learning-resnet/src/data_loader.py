from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ChestXrayDataset(Dataset):
    """
    PyTorch Dataset for the curated NIH Chest X-ray dataset.

    Each sample contains:
    - Image tensor
    - Binary classification label (0 for No Finding, 1 for Effusion)
    - Patient ID
    """
    def __init__(self, csv_path, image_dir, transform=None, split=None):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

        # Optionally select a particular dataset split
        if split is not None:
            self.df = self.df[self.df["Split"] == split].reset_index(drop=True)

        # Reset index for consistent indexing
        self.df = self.df.reset_index(drop=True) 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # Load image
        image_path = self.image_dir / row["Image Index"]
        with Image.open(image_path) as img:
            image = img.convert("L")
        # Apply preprocessing
        if self.transform is not None:
            image = self.transform(image)
        # Get label and patient ID
        label = int(row["Label"])
        patient_id = int(row["Patient ID"])
        return image, label, patient_id           
