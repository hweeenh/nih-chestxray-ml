# NIH Chest X-ray EDA using Pandas

## 1. Project Overview

This project explores the NIH Chest X-ray Dataset using Python and Pandas. The notebook reproduces and extends an earlier SQL-based exploratory data analysis (EDA), demonstrating how equivalent analyses can be performed using pandas while incorporating data cleaning, multi-label preprocessing, and clinical interpretation.

## 2. Dataset

NIH Chest X-ray dataset of 112,120 chest radiographs from 30,805 unique patients. There are 15 labelled findings categories, including a "No Finding", with multi-label annotations extracted from the radiology reports.

## 3. Objectives

- Explore patient demographics
- Investigate imaging characteristics
- Examine disease prevalence
- Analyse follow-up patterns
- Explore disease co-occurrence
- Practice data cleaning using pandas
- Compare SQL and pandas workflows

## 4. Skills demonstrated

### 4.1 Python
- pandas
- NumPy
### 4.2 Data Cleaning
- Missing value inspection
- Outlier detection
- Data validation
- Multi-label preprocessing
### 4.3 Exploratory Data Analysis
- GroupBy
- Aggregation
- Filtering
- Sorting
- Value counts
- Descriptive statistics
### 4.4 Medical Imaging
- Chest X-ray metadata
- AP vs PA projection
- Disease prevalence
- Multilabel disease interpretation
- Clinical interpretation

## 5. Exploratory Data Analysis

### 5.1 Dataset overview
Examined dataset structure, variables, patient identifiers, and disease label representation.
### 5.2 Data quality assessment
Inspected data for missing values, implausible values, duplicates, and inconsistency.
### 5.3 Age distribution
Analyzed patient age distribution and investigated potentially implausible ages.
16 patient records contained ages greater than 120. These observations were treated as implausible and excluded from age-related analyses rather than being removed from the entire dataset.
### 5.4 Gender distribution
Examined distribution of male and female demographic composition.
### 5.5 Disease prevalence
Analyzed frequency of individual disease findings across dataset.
### 5.6 View position analysis
Compared PA (posteroanterior) and AP (anteroposterior) radiographs. PA radiographs were more common than AP radiographs, in line with typical use of PA projection for ambulatory patients.
### 5.7 Follow-up patterns
Explored number of radiographs associated with individual patients to better understand repeated imaging and longitudinal nature of dataset.
### 5.8 Multi-label disease prevalence
The original disease annotation came with multiple findings within a single record. The labels were therefore separated to allow for prevalence and co-occurence analysis. 
### 5.9 Disease co-occurrence
Examined which radiological findings commonly appeared together, highlighting the multi-label nature of chest x-ray interpretation and provides motivation for future multi-label machine learning approaches.

## 6. Key Findings

- The dataset contains 112,120 chest radiographs from 30,805 patients.
- Sixteen patient ages greater than 120 years were identified as implausible and excluded from age-based analyses.
- PA radiographs were more common than AP radiographs, consistent with standard clinical practice.
- Splitting multi-label findings revealed substantially higher prevalences for Infiltration, Effusion, and Atelectasis, highlighting the importance of preprocessing multi-label data.
- Several radiological findings commonly occurred together, illustrating the multi-label nature of chest X-ray interpretation.

## 7. Repository Structure

.
├── NIH_Chest_Xray_EDA.ipynb
├── README.md

## 8. Future Improvements

### 8.1 Data Visualiation
- Visualize disease distributions with matplotlib
- Plot age and demographic distributions
- Compare disease prevalence across demographic groups
- Visualise AP vs PA distributions
### 8.2 Disease Co-occurrence
- Build disease co-occurrence heatmaps
- Investigate relationships between individual disease labels
### 8.3 Statistical Analysis
- Perform statistical hypothesis testing
- Investigate associations between demographic variables and disease findings
- Quantify relationships between demographic variables and disease prevalence
### 8.4 Machine Learning
- Develop baseline machine learning models
- Establish appropriate train, validation, and test splits at patient level
- Evaluate multi-label classification performance
- Investigate class imbalance
### 8.5 Deep Learning
- Develop CNN-based models for multi-label chest X-ray classification
- Explore medical imaging frameworks such as MONAI
