"""
# Script Overview
This script turns exploratory findings into a reproducible dataset preparation pipeline.

## Features
- Labels findings into 0 = No Finding, 1 = Effusion
- Splits train/validation patients into 5 folds, while grouping Patient ID, so images from 1 patient stay together
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# 1. Define paths and settings
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

CSV_PATH = DATA_DIR / "Data_Entry_2017_v2020.csv"
TRAIN_VAL_PATH = DATA_DIR / "train_val_list.txt"
TEST_PATH = DATA_DIR / "test_list.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

# 2. Load metadata and identify available images
df = pd.read_csv(CSV_PATH)

image_dir = DATA_DIR / "images"
available_images = {p.name for p in image_dir.glob("*.png")}

df = df[df["Image Index"].isin(available_images)].copy()

# 3. Select binary classification classes
has_effusion = df["Finding Labels"].str.split("|").apply(lambda labels: "Effusion" in labels)
is_no_finding = df["Finding Labels"] == "No Finding"

df = df[has_effusion | is_no_finding].copy()

df["Label"] = has_effusion[has_effusion | is_no_finding].astype(int)

# 4. Assign official test split
train_val_images = set(TRAIN_VAL_PATH.read_text().splitlines())
test_images = set(TEST_PATH.read_text().splitlines())

df["Split"] = df["Image Index"].apply(lambda name: "test" if name in test_images else "train_val")

# 5. Split train_val into train and validation while keeping patients together
train_val_df = df[df["Split"] == "train_val"].copy()
test_df = df[df["Split"] == "test"].copy()

splitter = StratifiedGroupKFold(
    n_splits = 5,
    shuffle=True,
    random_state=RANDOM_STATE)

train_indices, val_indices = next(splitter.split(
    train_val_df,
    y=train_val_df["Label"],
    groups = train_val_df["Patient ID"]))

train_df = train_val_df.iloc[train_indices].copy()
val_df = train_val_df.iloc[val_indices].copy()

train_df["Split"] = "train" 
val_df["Split"] = "val"

# 6. Combine and save processed metadata
final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
final_df = final_df[["Image Index", "Patient ID", "Label", "Split"]]

output_path = OUTPUT_DIR / "mlp_dataset.csv"
final_df.to_csv(output_path, index=False)

# 7. Print summary and check patient separation
print(f"Dataset split summary: {pd.crosstab(final_df["Split"], final_df["Label"])}")
print(f"Number of images: {final_df["Split"].value_counts()}")
print(f"Unique patients per split: {final_df.groupby("Split")["Patient ID"].nunique()}")

patient_sets = {
    split: set(group["Patient ID"])
    for split, group in final_df.groupby("Split")}

splits = ["train", "val", "test"]
for i, split_a in enumerate(splits):
    for split_b in splits[i+1:]:
        overlap = patient_sets[split_a] & patient_sets[split_b]
        print(f"Patient overlap ({split_a} vs {split_b}):", len(overlap))
print(f"Saved dataset metadata to: {output_path}")        