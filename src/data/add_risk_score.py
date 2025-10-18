# src/data/add_risk_score.py
import pandas as pd

# Load your processed IoT data
df = pd.read_csv("data/processed/iot_processed.csv")

# Simple formula for risk_score
df["risk_score"] = (
    (df["vibration"].abs()) * 0.5 +
    (df["load"].abs()) * 0.3 +
    (df["temp"] - df["temp"].mean()).abs() * 0.2
)

# Save back
df.to_csv("data/processed/iot_processed.csv", index=False)

print("✅ risk_score added and file updated!")
