# NIH Chest X-ray Image Classification

A deep learning project exploring binary chest X-ray classification using PyTorch and the NIH ChestX-ray14 dataset.

The project investigates the classification of chest X-ray images into two categories:

- **Effusion**
- **No Finding**

The main objective is to develop practical experience in medical image preprocessing, deep learning model development, and model evaluation, with particular attention to class imbalance in medical imaging datasets.

## Project Overview

This repository contains experiments using different approaches to image classification:

| Project | Description |
|---|---|
| [MLP](mlp/README.md) | A multilayer perceptron (MLP) baseline, with and without class weighting |
| [CNN](cnn/README.md) | Convolutional neural network experiments, including class weighting and image augmentation |
| [Tabular Baseline](tabular-baseline/README.md) | A metadata-only baseline for comparison |

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

**Dataset source:** [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-Nihil)

## Repository Structure

```text
nih-chestxray-ml/
├── src/
│   ├── data/
│   └── utils/
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
├── mlp/
│   ├── notebooks/
│   ├── results/
│   ├── src/
│   └── README.md
├── cnn/
│   ├── notebooks/
│   ├── src/
│   └── README.md
├── tabular-baseline/
│   └── README.md
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone <https://github.com/hweeenh/nih-chestxray-ml.git>
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

## Key Findings

The experiments explored the effect of class weighting and image augmentation on minority-class detection.

The weighted CNN achieved higher accuracy and precision than the weighted MLP on the held-out test set, while the weighted MLP achieved higher recall. Both models achieved similar F1-scores.

These findings highlight the importance of evaluating multiple performance metrics when working with imbalanced medical image datasets, rather than relying on accuracy alone.

## Limitations

- The experiments use a subset of the NIH ChestX-ray14 dataset.
- Images were resized to 64 × 64 pixels, which may result in the loss of clinically relevant detail.
- The classification task is limited to Effusion versus No Finding.
- The models have not been evaluated on an independent external dataset.

## Disclaimer

This project is for educational and portfolio purposes only. The models are experimental and are not intended for clinical use, diagnosis, or medical decision-making.