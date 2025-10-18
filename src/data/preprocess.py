import pandas as pd
import os

# Input & output paths
INPUT_PATH = "data/synthetic/iot_stream.csv"
OUTPUT_PATH = "data/processed/iot_processed.csv"

def preprocess():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input file not found at {INPUT_PATH}")

    # Load data
    df = pd.read_csv(INPUT_PATH)

    # Example preprocessing steps
    # Fill missing values
    df = df.fillna(method="ffill")

    # Normalize numerical columns
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    # Save processed data
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Preprocessed data saved to {OUTPUT_PATH} with {len(df)} rows")

if __name__ == "__main__":
    preprocess()
