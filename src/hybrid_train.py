# src/hybrid_train.py
import numpy as np
import joblib
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import os

def main():
    out_dir = "outputs"
    X_train = np.load(os.path.join(out_dir, "train_embeddings.npy"))
    y_train = np.load(os.path.join(out_dir, "train_labels.npy"))
    X_test = np.load(os.path.join(out_dir, "test_embeddings.npy"))
    y_test = np.load(os.path.join(out_dir, "test_labels.npy"))

    model = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Hybrid (LSTM embeddings -> XGB) Results")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/hybrid_xgb.pkl")
    print("Saved hybrid XGBoost at models/hybrid_xgb.pkl")

if __name__ == "__main__":
    main()
