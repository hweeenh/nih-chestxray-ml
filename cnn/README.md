# NIH Chest X-ray Classification — CNN

## Project overview

This project explores convolutional neural networks (CNNs) for binary classification of chest X-ray images from the NIH ChestX-ray14 dataset.

Building on the MLP baseline, this project investigates how convolutional layers extract spatial features from medical images and whether they improve classification performance.

The experiments cover:

- CNN architecture and convolutional feature extraction
- Image augmentation
- Class weighting to address class imbalance
- Training and validation loops using PyTorch
- Model evaluation using accuracy, precision, recall and F1-score

This is an educational machine-learning project. It is **not intended for clinical use or medical diagnosis**.

## Dataset

**Dataset:** NIH ChestX-ray14

The dataset was released by the National Institutes of Health (NIH).

### Files used

The CNN project uses the same dataset and preprocessing pipeline as the MLP baseline.

| File | Purpose |
|---|---|
| `Data_Entry_2017_v2020.csv` | Image labels and patient information |
| `train_val_list.txt` | Official training/validation image list |
| `test_list.txt` | Official test image list |
| `images_001.tar.gz` | Chest X-ray image archive (part 1) |

**Original dataset source:** [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737)

### Dataset scope and preprocessing

This project uses only the images available in `images_001.tar.gz`, which contains 4,999 images. From these, 3,241 images belonging to the two selected classes were used:

- No Finding: 2,754 images
- Effusion: 487 images

Images were converted to grayscale, resized to 64 × 64 pixels and normalised to the range 0–1.

The official test list was retained. The remaining training/validation images were split at patient level using stratified group splitting to prevent images from the same patient appearing across different splits.

The dataset files are not included in this GitHub repository because of their size. Refer to the root README for dataset download and setup instructions.

## Project structure

```text
cnn/
├── notebooks/
│   ├── 01_cnn_basics.ipynb
│   ├── 02_image_augmentation.ipynb
│   └── 03_cnn_model_development.ipynb
├── src/
│   └── cnn_model.py
└── README.md
```

The dataset preparation and image preprocessing scripts are shared at the repository level in `src/data/`.

## Notebooks

1. [CNN Basics](notebooks/01_cnn_basics.ipynb)
2. [Image Augmentation](notebooks/02_image_augmentation.ipynb)
3. [CNN Model Development](notebooks/03_cnn_model_development.ipynb)

## Environment and Dependencies

This project uses Python and PyTorch.

From the repository root, install the project dependencies using:

```bash
pip install -r requirements.txt
```

Refer to the root README for environment setup and dataset preparation instructions.

## Reproducibility

Experiments use a fixed random seed (42), seeded DataLoader shuffling, and deterministic cuDNN settings. The notebook was tested with Python 3.14.3 and the pinned dependencies in `requirements.txt`.

Exact results may vary across different hardware and software environments.

## CNN architecture

The CNN uses two convolutional layers followed by max pooling and fully connected classification layers.

The architecture consists of:

- First convolutional layer: 1 input channel → 16 feature maps
- ReLU activation
- Max pooling
- Second convolutional layer: 16 → 32 feature maps
- ReLU activation
- Max pooling
- Flattening
- Fully connected layer for binary classification

The model takes grayscale 64 × 64 images as input and outputs predictions for No Finding and Effusion.

## Experiments and evaluation

Three CNN configurations were investigated:

1. **Baseline CNN:** Trained without class weighting or image augmentation.
2. **Weighted CNN:** Uses class weighting in the loss function to address class imbalance.
3. **Augmented and weighted CNN:** Combines class weighting with image augmentation applied to the training data.

### Validation results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Baseline CNN | 85.92% | 38.10% | 23.53% | 29.09% |
| Weighted CNN | 82.31% | 35.85% | 55.88% | 43.68% |
| Augmented and weighted CNN | 78.88% | 33.33% | 72.06% | 45.58% |

The augmented and weighted CNN achieved the highest recall score and was selected for evaluation on the held-out test set.

### Held-out test results

| Metric | Augmented + Weighted CNN |
|---|---:|
| Accuracy | 58.04% |
| Precision | 41.18% |
| Recall | 79.33% |
| F1-score | 54.21% |

The selected augmented and weighted CNN model achieved an F1-score of 54.21% on the held-out test set, with 79.33% recall and 41.18% precision. It correctly identified 119 out of 150 Effusion cases, but also produced 170 false positives, indicating room for improvement in precision.

## Comparison with the MLP baseline

Both models were evaluated on the same held-out test set.

| Metric | Weighted MLP | Augmented + Weighted CNN |
|---|---:|---:|
| Accuracy | 51.77% | 58.04% |
| Precision | 37.84% | 41.18% |
| Recall | 84.00% | 79.33% |
| F1-score | 52.17% | 54.21% |

The MLP achieved higher recall while the CNN achieved higher accuracy and precision and F1-score.

The results do not demonstrate a clear overall performance advantage for the CNN. They highlight the importance of evaluating medical image classifiers using multiple metrics, particularly when class distributions are imbalanced.

## Limitations

- The experiments use only a subset of the NIH ChestX-ray14 dataset.
- Images were resized to 64 × 64 pixels, which may limit the preservation of fine anatomical details.
- The classification task is restricted to No Finding and Effusion.
- Class imbalance remains a challenge.
- The models have not undergone external validation and are not suitable for clinical application.

## Conclusion

This project provided practical experience in building and evaluating CNNs for medical image classification using PyTorch.

Compared with the MLP baseline, the CNN achieved higher accuracy and precision on the held-out test set, but did not achieve a higher F1-score or recall.

The experiments demonstrate how convolutional feature extraction, class weighting and image augmentation influence model performance, while highlighting the challenges of imbalanced medical imaging datasets.