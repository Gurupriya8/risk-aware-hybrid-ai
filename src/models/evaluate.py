import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Paths
INPUT_PATH = "data/processed/iot_processed.csv"
MODEL_DIR = "models"

def evaluate_models():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Processed data not found at {INPUT_PATH}")

    # Load data
    df = pd.read_csv(INPUT_PATH)
    X = df.drop(columns=["load", "timestamp", "device_id"])
    y = df["load"]

    # Define candidate models
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42)

    results = {}

    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, model in models.items():
        print(f"\n🚀 Training {name}...")
        model.fit(X, y)
        y_pred = model.predict(X)

        # Metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        results[name] = {"mse": mse, "r2": r2}
        print(f"📊 {name} -> MSE: {mse:.4f}, R²: {r2:.4f}")

        # Save model
        model_path = os.path.join(MODEL_DIR, f"{name}_model.pkl")
        joblib.dump(model, model_path)
        print(f"💾 Saved {name} model at {model_path}")

        # Feature Importance (only if available)
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            feature_names = X.columns

            plt.figure(figsize=(8, 5))
            plt.barh(feature_names, importance, color="skyblue")
            plt.xlabel("Importance")
            plt.title(f"Feature Importance ({name})")
            plt.tight_layout()

            plot_path = os.path.join(MODEL_DIR, f"{name}_feature_importance.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"📈 Saved {name} feature importance plot at {plot_path}")

    print("\n✅ Evaluation complete. Summary:")
    for name, metrics in results.items():
        print(f"{name}: MSE={metrics['mse']:.4f}, R²={metrics['r2']:.4f}")

if __name__ == "__main__":
    evaluate_models()
