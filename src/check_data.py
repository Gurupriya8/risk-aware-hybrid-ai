import pandas as pd

# load your dataset
df = pd.read_csv("data/processed/iot_processed.csv")


# show the first few rows
print("Column names:")
print(df.columns)
print("\nFirst 5 rows:")
print(df.head())
