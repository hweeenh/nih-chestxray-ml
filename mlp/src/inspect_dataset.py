"""
# Script Overview
This script is an exploratory script using python

## Features
- Reads input from "Data_Entry_2018_v2020.csv" and extracted "images_001.tar.gz"
- Maps multi-label strings into binary format, isolating a dual-class subset of Effusion and No Finding
- Categorize each image into its respective official splits based on reference text files
- Tracks class distibution across splits
- Audits patient isolation by calculating intersection of Patient ID sets between partition lists to guarantee zero-patient overlaps
"""

from pathlib import Path
import pandas as pd

# Project paths
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
CSV_PATH = DATA_DIR / "Data_Entry_2017_v2020.csv"

# Load dataset metadata
df = pd.read_csv(CSV_PATH)

# Basic information
print("Dataset shape:", df.shape)
print(f"Column names: {df.columns.tolist()}")

# Count images available in extracted archive
image_dir = DATA_DIR / "images"
image_files = {p.name for p in image_dir.glob("*.png")}
df_subset = df[df["Image Index"].isin(image_files)].copy()
print("Images in metadata:", len(df))
print("Images extracted:", len(image_files))
print("Images matched to metadata:", len(df_subset))

# Count labels in extracted subset
print(f"Finding label distribution: {df_subset["Finding Labels"].value_counts().head(20)}")

# Count target classes
effusion_count = df_subset[df_subset["Finding Labels"].str.split("|").apply(lambda labels: "Effusion" in labels)].shape[0]
no_finding_count = df_subset[df_subset["Finding Labels"] == "No Finding"].shape[0]
print("Target label counts:")
print("Images containing Effusion:", effusion_count)
print("Images labelled No Finding:", no_finding_count)


# Check official train/validation and test lists
TRAIN_VAL_PATH = DATA_DIR / "train_val_list.txt"
TEST_PATH = DATA_DIR / "test_list.txt"

train_val_images = set(TRAIN_VAL_PATH.read_text().splitlines())
test_images = set(TEST_PATH.read_text().splitlines())

# Select our 2 target classes
has_effusion = df_subset["Finding Labels"].str.split("|").apply(lambda labels: "Effusion" in labels)
is_no_finding = df_subset["Finding Labels"] == "No Finding"
target_df = df_subset[has_effusion | is_no_finding].copy()

# Assign binary labels
target_df["Label"] = has_effusion[has_effusion | is_no_finding].astype(int)

# Identify which official split each image belongs to
target_df["Split"] = target_df["Image Index"].apply(lambda name: (
    "train_val" if name in train_val_images
    else "test" if name in test_images
    else "unlisted"))
print(f"Target dataset split distribution: {target_df["Split"].value_counts()}")
print(f"Class distribution by split: {pd.crosstab(target_df["Split"], target_df["Label"])}")

# Check whether patients appear in both official lists (Should have 0 patient overlap)
train_val_patients = set(df[df["Image Index"].isin(train_val_images)]["Patient ID"])
test_patients = set(df[df["Image Index"].isin(test_images)]["Patient ID"])
print("Patients in both official lists:", len(train_val_patients & test_patients))
print("Unique patients in target dataset:", target_df["Patient ID"].nunique())