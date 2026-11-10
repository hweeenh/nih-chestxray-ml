# NIH Chest X-ray Image Classification

A deep learning project exploring binary chest X-ray classification using PyTorch and the NIH ChestX-ray14 dataset.

The project investigates the classification of chest X-ray images into two categories:

- **Effusion**
- **No Finding**

The main objective is to develop practical experience in medical image preprocessing, deep learning model development, and model evaluation, with particular attention to class imbalance in medical imaging datasets.

## Project Overview

This repository explores binary chest X-ray classification using deep learning and machine learning approaches.

The CNN is the primary project, with MLP and metadata-only experiments serving as supporting baselines.

| Project | Description |
|---|---|
| [CNN — Main Project](cnn/README.md) | CNN architecture, image augmentation, class weighting and evaluation |
| [MLP Baseline](mlp/README.md) | Multilayer perceptron baseline with and without class weighting |
| [Tabular Baseline](tabular-baseline/README.md) | Metadata-only classification baseline |

The MLP and CNN experiments use the same preprocessed images and patient-level training, validation, and test splits, allowing their performance to be compared on a consistent dataset.

## Dataset

This project uses a subset of the **NIH ChestX-ray14 dataset**, specifically images obtained from the `images_001.tar.gz` archive and the associated metadata and official split files.

The binary classification task focuses on:

- **No Finding:** 2,754 images
- **Effusion:** 487 images
- **Total:** 3,241 images

The selected images were converted to grayscale, resized to 64 × 64 pixels, and normalised to a range of 0–1.

Patient-level splitting was used to separate the training, validation, and test datasets:

| Split | Images |
|---|---:|
| Training | 2,208 |
| Validation | 554 |
| Test | 479 |
| **Total** | **3,241** |

The original images, metadata, split files, and processed datasets are not included in this repository due to their size. Users should obtain the required files directly from the official NIH ChestX-ray14 dataset source and follow the preparation instructions in this repository.

**Dataset source:** [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)

## Repository Structure

nih-chestxray-ml/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── data/
│   │   ├── inspect_dataset.py
│   │   ├── inspect_images.py
│   │   ├── visualize_samples.py
│   │   ├── preprocess_images.py
│   │   └── prepare_dataset.py
│   │
│   └── utils/
│
├── cnn/
│   ├── README.md
│   ├── notebooks/
│   │   ├── 01_cnn_basics.ipynb
│   │   ├── 02_image_augmentation.ipynb
│   │   └── 03_cnn_model_development.ipynb
│   └── src/
│       └── cnn_model.py
│
├── mlp/
│   ├── README.md
│   ├── notebooks/
│   │   └── 01_mlp_model_development.ipynb
│   └── src/
│       └── mlp_model.py
│
├── tabular-baseline/
│   ├── README.md
│   └── nih_cxr_tabular_baseline.ipynb
│
├── pandas-eda/
│   ├── README.md
│   └── NIH_Chestxray14_Pandas_EDA.ipynb
│
└── sql-eda/
    ├── README.md
    └── eda.py

**Note:** The `data/` directory is excluded from Git and is not included in this repository. It is created locally when setting up the project and contains the downloaded raw NIH ChestX-ray14 files, metadata, and processed datasets. Refer to the Dataset and Getting Started sections for instructions on obtaining and preparing the required files.  

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/hweeenh/nih-chestxray-ml.git
cd nih-chestxray-ml
```

### 2. Set up the Python environment

```bash
python -m venv .venv
```

Activate the environment:

**macOS / Linux**

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### 3. Obtain and prepare the dataset

Download the NIH ChestX-ray14 dataset from the source linked above.

Follow the data preparation instructions to:

1. Place the original dataset files in `data/raw/`.
2. Prepare the metadata and patient-level dataset splits.
3. Preprocess the selected images.
4. Generate the processed image tensors required by the MLP and CNN notebooks.

Detailed commands and file requirements will be documented once the data preparation workflow has been verified.

### 4. Run the experiments

The main experiment notebooks are:

- **MLP:** `mlp/notebooks/01_mlp_model_development.ipynb`
- **CNN:** `cnn/notebooks/03_cnn_model_development.ipynb`

Open the notebooks in VS Code or Jupyter and execute the cells in order after completing the dataset preparation steps.

Refer to each project's README for its architecture, experiments, evaluation results, and specific instructions.

## Experiments

- [CNN — Main Project](cnn/README.md)
- [MLP Baseline](mlp/README.md)
- [Tabular Baseline](tabular-baseline/README.md)

## Key Findings

The experiments explored the effect of class weighting and image augmentation on minority-class detection.

The augmented and weighted CNN achieved higher accuracy, precision, and F1-score than the weighted MLP on the held-out test set, while the weighted MLP achieved higher recall.

These findings highlight the importance of evaluating multiple performance metrics when working with imbalanced medical image datasets, rather than relying on accuracy alone.

## Limitations

- The experiments use a subset of the NIH ChestX-ray14 dataset.
- Images were resized to 64 × 64 pixels, which may result in the loss of clinically relevant detail.
- The classification task is limited to Effusion versus No Finding.
- The models have not been evaluated on an independent external dataset.

## Disclaimer

This project is for educational and portfolio purposes only. The models are experimental and are not intended for clinical use, diagnosis, or medical decision-making.