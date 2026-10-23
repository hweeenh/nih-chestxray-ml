# NIH Chest X-ray Dataset - SQL Exploratory Data Analysis

# What is this project?
This project uses SQL to explore NIH Chest X-ray(CXR) dataset, which contains 112,120 CXR records from 30,805 unique patients. The goal is to understand patterns in the data before building machine learning models for medical imaging.

# Why this dataset?
I am a radiographer pivoting into AI for medical imaging. This dataset is directly relevant to my clinical background, and my goal of applying machine learning to CXR analysis. X-rays are suitable as a first-line imaging modality due to its portability, cost effectiveness, and quick initial diagnostic capability. Chest X-rays are the most common type of x-rays performed as it shows images of the heart, lungs, airways, blood vessels, and bones of the thoracic cage.

# What questions did I explore?
1. How many unique patients and scans are in the dataset?
2. What is the average patient age by X-ray view position (PA vs AP)?
3. What is the gender distribuion of patients?
4. Which disease findings are most common?
5. Which patients had the most follow-up scans?
6. What are the most common findings by view position?
7. What is the average age and number of occurence by disease?
8. Findings by follow-up number.
9. What is the most common disease, accounting for disease combinations?
10. Which diseases commonly co-occur? What is the rate of the co-occurrence?

# Key findings
1. This dataset contains 112,120 scans from 30,805 unique patients.
2. PA (Posteroanterior) view accounts for 67,310 scans while AP (Anteroposterior) view accounts for 44,810 scans. This is consistent with clinical practice that PA view is the standard, optimal view for ambulatory patients, to reduce the enlarging effect AP view has on the heart.
3. The dataset contains scans from 48,780 females and 63,340 males, accounting for 43.51% and 56.49% respectively.
4. "No Finding" is the most common label (60,361 counts), indicating significant class imbalance that will need to be addressed when building classifiers.
5. "Infiltration" is the most common disease finding (9.547 scans), followed by Atelectasis and Effusion.
6. Patient 10007 has 184 follow-up scans - indicative of a chronic condition being monitored over time.
7. Most common co-occurring diseases are "Effusion | Infiltration", accounting for 1,603 scans. Followed by "Atelectasis | Infiltration" (1,350 scans), and "Atelectasis | Effusion (1,165 scans). Disease co-occurrence is clinically significant -- For example, effusion with infiltration could suggest pneumonia with pleural involvement. Cardiomegaly|Edema|Effusion could suggest congestive heart failure. Nodule|Mass|Atelectasis|Effusion could suggest lung cancer. Emphysema|Pneumothorax may suggest COPD. 

# Tools used
- Python 3
- SQLite
- pandas
- VSCode

# How to run this project
1. Clone this repository
2. Download Data_Entry_2017.csv from the NIH Chest X-ray dataset on Kaggle (https://www.kaggle.com/datasets/nih-chest-xrays/data) and place it in project folder.
3. Run 'python3 eda.py'

# About me
I am a radiographer in Singapore, specialising in x-ray and ultrasound. I am pivoting into AI for medical imaging. This project is part of my preparation for AI Programme and training. 
