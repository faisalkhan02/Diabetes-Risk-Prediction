
import pandas as pd

# Load the Kaggle dataset
df = pd.read_csv("data/diabetes.csv")

# Display the first five rows
print("First five rows:")
print(df.head())

# Display dataset dimensions
print("\nDataset shape:")
print(df.shape)

# Display column names
print("\nColumn names:")
print(df.columns.tolist())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())



print("\nDataset information:")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))

print("\nMissing value percentage:")
missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent[missing_percent > 0].round(2))

print("\nTarget column distribution:")
print(df["diabetes_risk"].value_counts(dropna=False))

print("\nTarget column data type:")
print(df["diabetes_risk"].dtype)



print("\nColumn data types:")
print(df.dtypes)

print("\nUnique values in categorical columns:")
categorical_cols = df.select_dtypes(include=["object", "category"]).columns

for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False).head(10))

print("\nNumerical summary:")
print(df.describe())


