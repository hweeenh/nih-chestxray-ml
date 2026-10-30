# NIH Chest X-ray Classification — MLP Baseline

## Project overview

This project explores a simple **multilayer perceptron (MLP)** for binary classification of chest X-ray images from the NIH ChestX-ray14 dataset.

The aim is to understand the fundamentals of image classification using PyTorch, including:
- Image loading and preprocessing
- Building an MLP architecture
- Training and validation loops
- Model evaluation

This is an educational machine-learning project. It is **not intended for clinical use or medical diagnosis**.

## Dataset

**Dataset:** NIH ChestX-ray14

The dataset was released by the National Institutes of Health (NIH).

### Files used

Download the following files from the original dataset source:
https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737

| File | Purpose |
|---|---|
| `Data_Entry_2017_v2020.csv` | Image labels and patient information |
| `train_val_list.txt` | Official training/validation image list |
| `test_list.txt` | Official test image list |
| `images_001.tar.gz` | Chest X-ray image archive (part 1) |

**Original dataset source:** [Insert the exact webpage or download link you used]

### Dataset setup

After downloading, place the files in:

```text
mlp/
└── data/
    └── raw/
        ├── Data_Entry_2017_v2020.csv
        ├── train_val_list.txt
        ├── test_list.txt
        └── images_001.tar.gz
```

Extract the image archive inside `data/raw/`:

```bash
tar -xzf data/raw/images_001.tar.gz -C data/raw/
```

The extracted images should be located in:

```text
data/raw/images/
```

**Note:** The dataset files are not included in this GitHub repository because of their size. Users should download them from the original source and follow its terms of use.

## Project structure

```text
mlp/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   └── inspect_dataset.py
├── results/
└── README.md
```

## Environment and dependencies

This project uses Python and PyTorch.

The exact package versions and installation instructions will be documented as the project develops.

## Current progress

- [x] Downloaded dataset metadata and initial image archive
- [x] Created project folder structure
- [x] Created dataset inspection script
- [ ] Inspect dataset labels and class distribution
- [ ] Prepare binary classification dataset
- [ ] Create patient-level training, validation and test splits
- [ ] Build and train MLP model
- [ ] Evaluate model performance

## Learning objectives

This project is part of my learning journey into deep learning and medical imaging AI. It serves as a baseline before exploring convolutional neural networks (CNNs) and transfer learning for chest X-ray classification.