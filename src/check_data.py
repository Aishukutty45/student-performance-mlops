import pandas as pd

# Load dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

print("========== DATA VALIDATION ==========")

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Columns
print("\nColumns:")
print(df.columns.tolist())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Target information
print("\nTarget Variable: G3")
print("Minimum:", df["G3"].min())
print("Maximum:", df["G3"].max())
print("Average:", df["G3"].mean())

# Data types
print("\nData Types:")
print(df.dtypes)

# Validation
if df.isnull().sum().sum() == 0:
    print("\nMissing value check: PASSED")
else:
    print("\nMissing value check: FAILED")

if df.duplicated().sum() == 0:
    print("Duplicate check: PASSED")
else:
    print("Duplicate check: FAILED")

if "G3" in df.columns:
    print("Target column check: PASSED")
else:
    print("Target column check: FAILED")

print("\n========== VALIDATION COMPLETED ==========")