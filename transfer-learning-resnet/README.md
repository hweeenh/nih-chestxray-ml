# Chest X-ray Classification: Pleural Effusion Detection Using ResNet18

## Project Overview

This project explores the use of deep learning for binary classification of pleural effusion in frontal chest X-ray images using the NIH Chest X-ray dataset.

Two convolutional neural network approaches were implemented and evaluated:

1. **Scratch CNN:** A convolutional neural network trained from randomly initialised weights.
2. **ResNet18 Transfer Learning:** An ImageNet-pretrained ResNet18 fine-tuned for chest X-ray classification.

The project investigates whether transfer learning can improve classification performance compared with training a CNN from scratch, while also examining model behaviour through Grad-CAM visualisations.

An additional motivation was to explore how my background in diagnostic radiography could complement technical skills in medical imaging AI.

## Clinical Context

Pleural effusion refers to the accumulation of fluid in the pleural space. On chest radiographs, it may manifest through findings such as costophrenic angle blunting and basal opacification.

Automated identification of radiographic findings is an area of interest in medical imaging AI. However, good classification performance alone does not establish clinical reliability. Understanding the underlying anatomy, radiographic appearances and limitations of model interpretation remains important when evaluating these systems.

My background in diagnostic radiography provided a foundation for understanding chest X-ray anatomy, AP and PA projections, image characteristics and the clinical context of radiographic findings.

## Dataset

**Dataset source:** [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)

A curated subset of 3,241 frontal chest X-ray images from 1,127 unique patients was used.

The binary classification task was defined as:

- **Positive:** Images containing the exact finding label `Effusion`, including images with additional findings.
- **Negative:** Images labelled `No Finding`.
- Images with other findings but without Effusion were excluded.

### Dataset Splits

Patient-level separation was maintained to prevent images from the same patient appearing across different splits.

| Split | Images | Negative | Positive |
|---|---:|---:|---:|
| Training | 2,208 | 1,939 | 269 |
| Validation | 554 | 486 | 68 |
| Test | 479 | 329 | 150 |
| **Total** | **3,241** | **2,754** | **487** |

The official test list was retained for the held-out test set. The remaining official training/validation data was divided using `StratifiedGroupKFold` with five folds, shuffling and random state 42.

The dataset is imbalanced, with Effusion-positive images representing approximately 12.2% of the training set.

The test set also has a different class prevalence and AP/PA projection distribution from the training and validation sets. These differences are considered when interpreting model performance.

## Methodology

### 1. Data Exploration and Preprocessing

Exploratory data analysis was performed to examine:

- Class distribution and imbalance.
- Patient distribution across dataset splits.
- AP and PA projection composition.
- Original image dimensions, format and pixel intensity characteristics.

All images were originally 1024 × 1024 grayscale images.

Preprocessing included:

- Resizing to 224 × 224 pixels.
- Converting grayscale images into three channels for compatibility with pretrained ResNet18.
- Converting images into PyTorch tensors.
- Applying ImageNet normalisation.

The preprocessing pipeline and custom PyTorch Dataset were implemented in reusable Python modules under `src/`.

### 2. CNN Baseline

A simple three-convolution-layer CNN was developed and trained from random initialisation.

This model served as a baseline to evaluate the performance of a CNN without pretrained weights.

Class imbalance was addressed using a positive class weight in `BCEWithLogitsLoss`.

### 3. ResNet18 Transfer Learning

An ImageNet-pretrained ResNet18 was adapted for binary classification by replacing its final fully connected layer with a single-output classification layer.

Training was performed in two stages:

- **Stage 1:** The pretrained backbone was frozen while the classification head was trained.
- **Stage 2:** The entire network was unfrozen and fine-tuned using differential learning rates for the backbone and classification head.

The best checkpoint was selected based on validation ROC-AUC.

### 4. Model Evaluation

Both models were evaluated using the same held-out test set.

Evaluation metrics included:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision

Classification thresholds were selected using validation data by maximising F1-score, then applied unchanged to the test set.

Confusion matrices, ROC curves and precision-recall curves were used to examine model performance.

### 5. Model Interpretability

Grad-CAM was applied to selected ResNet18 predictions to investigate the image regions contributing to the model's positive-class predictions.

Two true-positive and two false-negative examples, each from a different patient, were examined.

The visualisations provided an opportunity to assess whether the model highlighted anatomically plausible regions. Some examples showed attention around the lung bases, while others highlighted less relevant structures or medical devices.

These observations highlight the importance of evaluating model behaviour beyond quantitative classification metrics.

## Results

### Test Set Performance

| Metric | Scratch CNN | ResNet18 |
|---|---:|---:|
| Accuracy | 0.5532 | **0.7328** |
| Precision | 0.3644 | **0.5579** |
| Recall | 0.5733 | **0.7067** |
| F1-score | 0.4456 | **0.6235** |
| ROC-AUC | 0.5722 | **0.7927** |
| Average Precision | 0.4147 | **0.5895** |

ResNet18 outperformed the scratch CNN across all six evaluation metrics.

The improvement was also reflected in the confusion matrices:

| | Scratch CNN | ResNet18 |
|---|---:|---:|
| True Negatives | 179 | 245 |
| False Positives | 150 | 84 |
| False Negatives | 64 | 44 |
| True Positives | 86 | 106 |

Despite the improvement, ResNet18 still failed to identify 44 of the 150 Effusion-positive test images, corresponding to a recall of 70.7%.

These results demonstrate the benefit of transfer learning for this classification task, but do not establish clinical reliability.

## Limitations

Several limitations should be considered:

- The dataset represents a relatively small, curated subset of the original NIH Chest X-ray dataset.
- The held-out test set contains only 52 unique patients, with multiple images contributed by many patients.
- Differences in class prevalence and AP/PA projection distribution may affect generalisability.
- The classification task distinguishes Effusion-positive images from No Finding images; it does not localise effusions or establish their clinical significance.
- Grad-CAM provides coarse visual explanations and does not prove that the model relies on clinically meaningful features.
- Performance on this dataset does not establish generalisability to other institutions, patient populations or imaging systems.

## Conclusion and Key Learning

This project provided an opportunity to apply deep learning techniques to a medical imaging classification problem while drawing on my existing background in diagnostic radiography.

My clinical knowledge helped me understand the radiographic data, recognise relevant anatomical structures and critically assess the plausibility of model visualisations. This highlighted how clinical understanding can complement technical evaluation when developing and examining medical imaging AI systems.

The project also reinforced an important distinction between quantitative model performance and clinically meaningful interpretation. Although ResNet18 demonstrated better classification performance than the scratch CNN, its visualisations showed that predictions may still be influenced by incidental or less relevant image features.

Overall, this project represents an initial step towards bridging my experience in diagnostic radiography with technical capabilities in machine learning and medical imaging AI.

## Project Structure

```text
transfer-learning-resnet/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_resnet_basics.ipynb
│   ├── 02_data_exploration.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_scratch_cnn.ipynb
│   ├── 05_resnet18_transfer_learning.ipynb
│   └── 06_model_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── transforms.py
│   └── models.py
├── results/
└── models/
```

## Running the Project

### Development Environment

The project was developed using:
- Python 3.14.3
- PyTorch 2.15.0
- Torchvision 0.29.2
- Apple Silicon Mac with MPS acceleration

Package dependencies and versions are documented in 'requirements.txt'.

Note: Training reproducibility may vary across hardward and software configurations.

### Requirements

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Pillow
- Jupyter Notebook

Install the project dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Setup

The NIH Chest X-ray dataset must be obtained separately.

The project expects the dataset to be located in a `data/` directory alongside the `transfer-learning-resnet/` project directory, following the paths used in the notebooks.

The dataset and patient-level data files are not included in this repository.

### Execution Order

For the complete workflow, execute the notebooks in numerical order:

1. `01_resnet_basics.ipynb`
2. `02_data_exploration.ipynb`
3. `03_data_preprocessing.ipynb`
4. `04_scratch_cnn.ipynb`
5. `05_resnet18_transfer_learning.ipynb`
6. `06_model_evaluation.ipynb`

The model evaluation notebook loads the checkpoints generated by the two training notebooks.

## Dataset Source and Acknowledgement

The dataset used in this project is the [NIH ChestX-ray14 dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC).

**Original publication:**

Wang, X. et al. (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

The project was developed for educational and portfolio purposes. It is not intended for clinical diagnosis or direct patient-care decisions.