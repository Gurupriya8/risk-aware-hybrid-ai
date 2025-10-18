import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Paths
INPUT_PATH = "data/processed/iot_processed.csv"
MODEL_PATH = "models/risk_model.pkl"

def train_model():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Processed file not found at {INPUT_PATH}")

    # Load processed dataset
    df = pd.read_csv(INPUT_PATH)

    # Features and target
    X = df.drop(columns=["load", "timestamp", "device_id"])  # drop non-numeric/categorical
    y = df["load"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model (Random Forest Regressor for resilience forecasting)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"✅ Model trained")
    print(f"📊 MSE: {mse:.4f}, R²: {r2:.4f}")

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved at {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
