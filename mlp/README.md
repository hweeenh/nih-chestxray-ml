# NIH Chest X-ray Classification — MLP Baseline

## Project overview

This project explores a **multilayer perceptron (MLP)** for binary classification of chest X-ray images from the NIH ChestX-ray14 dataset.

The aim is to understand the fundamentals of image classification using PyTorch, including:

- Image loading and preprocessing
- Building an MLP architecture
- Training and validation loops
- Handling class imbalance
- Model evaluation

This is an educational machine-learning project. It is **not intended for clinical use or medical diagnosis**.

## Dataset

**Dataset:** NIH ChestX-ray14

The dataset was released by the National Institutes of Health (NIH).

### Files used

Download the following files from the original dataset source:

| File | Purpose |
|---|---|
| `Data_Entry_2017_v2020.csv` | Image labels and patient information |
| `train_val_list.txt` | Official training/validation image list |
| `test_list.txt` | Official test image list |
| `images_001.tar.gz` | Chest X-ray image archive (part 1) |

**Original dataset source:** [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737)

### Dataset setup

Download the files and place them in the following locations relative to the repository root:

```text
data/
├── raw/
│   ├── Data_Entry_2017_v2020.csv
│   └── images_001.tar.gz
└── metadata/
    ├── train_val_list.txt
    └── test_list.txt
```

Extract the image archive from the repository root:

```bash
tar -xzf data/raw/images_001.tar.gz -C data/raw/
```

The extracted images should be located in:

```text
data/raw/images/
```

**Dataset scope:** This project uses only the images available in `images_001.tar.gz`, which contains 4,999 images. From these, 3,241 images belonging to the two selected classes were used:

- No Finding: 2,754 images
- Effusion: 487 images

The dataset files are not included in this GitHub repository because of their size. Users should download them from the original source and follow its terms of use.

## Project structure

```text
mlp/
├── notebooks/
│   └── 01_mlp_model_development.ipynb
├── src/
│   └── mlp_model.py
├── results/
└── README.md
```

The shared dataset preparation and image preprocessing scripts are located in the repository-level `src/data/` directory.

## Environment and dependencies

This project uses Python and PyTorch.

From the repository root, install the project dependencies using:

```bash
pip install -r requirements.txt
```

See the root `README.md` for environment setup and dataset preparation instructions.

## Experiments and evaluation

Three MLP evaluations were conducted:

| Model | Evaluation set | Accuracy | Precision | Recall | F1-score |
|---|---|---:|---:|---:|---:|
| Baseline MLP | Validation | 87.55% | 0.00% | 0.00% | 0.00% |
| Weighted MLP | Validation | 72.02% | 28.78% | 86.76% | 43.22% |
| Weighted MLP | Test | 51.77% | 37.84% | 84.00% | 52.17% |

Class weighting improved the model's ability to identify Effusion cases, increasing recall substantially compared with the baseline. However, this came at the expense of overall accuracy and precision.

The results highlight the limitations of a simple MLP for chest X-ray classification, particularly when working with imbalanced medical imaging datasets.

The weighted MLP was subsequently compared against a CNN using the same held-out test set.

## Learning objectives

This project is part of my learning journey into deep learning and medical imaging AI. It provides a foundation in neural network training, model evaluation and class imbalance before exploring convolutional neural networks (CNNs) and transfer learning for chest X-ray classification.