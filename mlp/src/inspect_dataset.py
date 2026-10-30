import pandas as pd

# Load dataset metadata
df = pd.read_csv("data/raw/Data_Entry_2017_v2020.csv")

# Basic information
print("Dataset shape:", df.shape)
print(f"Column names: {df.columns.tolist()}")
print(f"First 5 rows: {df.head()}")
print(f"Number of unique patients: {df["Patient ID"].nunique()}")
print(f"Finding label distribution: {df["Finding Labels"].value_counts().head(20)}")