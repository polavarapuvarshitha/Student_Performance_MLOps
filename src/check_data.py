import pandas as pd

df = pd.read_csv("data/StudentsPerformance.csv")

print("Dataset loaded successfully!")
print()

print("Shape:")
print(df.shape)

print()

print("Columns:")
print(df.columns.tolist())

print()

print("First 5 rows:")
print(df.head())

print()

print("Missing values:")
print(df.isnull().sum())