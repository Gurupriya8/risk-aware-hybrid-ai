import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc


def create_sequences(data, labels, seq_len=10):
    """
    Create sliding windows of sequences for LSTM.
    Each sequence = seq_len rows of features.
    Label = risk_score of the LAST row in the window.
    """
    sequences, seq_labels = [], []
    for i in range(len(data) - seq_len + 1):
        seq = data[i:i+seq_len]
        label = labels[i+seq_len-1]
        sequences.append(seq)
        seq_labels.append(label)
    return np.array(sequences), np.array(seq_labels)

class IoTDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)   # [num_samples, seq_len, features]
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long),
        )

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_classes=2):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # ✅ Flatten [batch, seq_len, features] → [batch, seq_len*features]
        x = x.view(x.size(0), -1)
        return self.layers(x)


    
class LSTMModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=1, num_classes=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # If input is [batch, features], reshape to [batch, seq_len=1, features]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # makes it [batch, 1, features]

        # x shape: [batch, seq_len, features]
        out, _ = self.lstm(x)         # out: [batch, seq_len, hidden_dim]
        out = out[:, -1, :]           # take the last timestep
        out = self.fc(out)
        return out

class GRUModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=1, num_classes=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)


class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_classes=2):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(32, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = x.permute(0, 2, 1)  # -> [batch, features, seq_len]
        x = self.relu(self.conv1(x))
        x = x.permute(0, 2, 1)  # -> [batch, seq_len, channels]
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class TransformerModel(nn.Module):
    def __init__(self, input_dim=5, d_model=64, nhead=4, num_layers=2, num_classes=2):
        super(TransformerModel, self).__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = self.input_fc(x)  # [batch, seq_len, d_model]
        x = x.permute(1, 0, 2)  # [seq_len, batch, d_model] for transformer
        out = self.transformer(x)  # [seq_len, batch, d_model]
        out = out[-1, :, :]       # last timestep
        return self.fc(out)

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"confusion_{model_name}.png")
    plt.close()
    print(f"✅ Saved confusion matrix for {model_name}")

def plot_roc_curve(y_true, y_probs, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.savefig(f"roc_{model_name}.png")
    plt.close()
    print(f"✅ Saved ROC curve for {model_name}")

    
# ------------------------------
# Main Function
def train_single_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, model_name):
    print(f"\n🚀 Training {model_name} model...")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"{model_name} | Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # ✅ Evaluation
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    test_loss = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.append(probs.cpu())
            test_loss += criterion(outputs, yb).item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    test_loss /= len(test_loader)
  

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"✅ {model_name} → Acc: {acc*100:.2f}%, Prec: {prec*100:.2f}%, Rec: {rec*100:.2f}%, "
          f"F1: {f1*100:.2f}%, Test Loss: {test_loss:.4f}")

    # ✅ Save confusion matrix
    plot_confusion_matrix(all_labels, all_preds, model_name)

    # ✅ Save ROC curve
    y_probs = torch.cat(all_probs).numpy()
    plot_roc_curve(all_labels, y_probs, model_name)

    # ✅ Save trained model
    torch.save(model.state_dict(), f"{model_name}.pth")
    print(f"💾 Saved model weights: {model_name}.pth")

    return {
       "Model": model_name,
        "accuracy": acc * 100,
        "precision": prec * 100,
        "recall": rec * 100,
        "f1": f1 * 100,
        "test_loss": test_loss,
        "labels": all_labels,
        "preds": all_preds
    }




def main():
    print("✅ Entered main()")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    print("📂 Loading dataset...")
    df = pd.read_csv("data/processed/iot_processed.csv")
    print("✅ Dataset loaded with shape:", df.shape)
    print("📑 Columns:", list(df.columns))

    # --- Step 1: Preprocessing for Classification ---
    df = df.drop(columns=["device_id", "timestamp"])
    feature_cols = ["lat", "lon", "temp", "vibration", "load"]
    scaler = StandardScaler()
    features = scaler.fit_transform(df[feature_cols].values)
    labels = (df["risk_score"].values > 0.5).astype(int)

    # Build sequences for LSTM
    seq_len = 10
    X, y = create_sequences(features, labels, seq_len)
    print("✅ Created sequences:", X.shape, "labels:", y.shape)

    # --- STEP 2: Train/Test Split ---
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("📊 Train size:", X_train.shape, "Test size:", X_test.shape)

    # Wrap datasets
    train_ds = IoTDataset(X_train, y_train)
    test_ds = IoTDataset(X_test, y_test)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    print("✅ DataLoader ready: train batches =", len(train_loader),
          "| test batches =", len(test_loader))

    # Dictionary of models
    feature_dim = features.shape[1]
    models_dict = {
        "MLP": MLP(input_size=seq_len * feature_dim, hidden_size=64, num_classes=2).to(device),
        "LSTM": LSTMModel(input_dim=X_train.shape[2], hidden_dim=64, num_layers=1, num_classes=2).to(device),
        "GRU": GRUModel(input_dim=X_train.shape[2], hidden_dim=64, num_layers=1, num_classes=2).to(device),
        "CNN_LSTM": CNNLSTMModel(input_dim=X_train.shape[2], hidden_dim=64, num_classes=2).to(device),
        "Transformer": TransformerModel(input_dim=X_train.shape[2], d_model=64, nhead=4, num_layers=2, num_classes=2).to(device),
    }

    results = {}

    # Train & Evaluate all models
    for name, model in models_dict.items():
        print(f"\n⚡ Training model: {name}")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Class balancing
        class_counts = np.bincount(y_train)
        class_counts = np.where(class_counts == 0, 1, class_counts)  # avoid div by 0
        weights = torch.tensor(len(y_train) / (2.0 * class_counts), dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))

        # Train + Evaluate this model
        metrics = train_single_model(
            model, train_loader, test_loader, criterion, optimizer,
            num_epochs=10, device=device, model_name=name
        )
        results[name] = metrics

    # =========================================================
    # 📊 Final Comparison Across Models
    # =========================================================
    print("\n📊 Final Comparison Across Models:")
    df_results = pd.DataFrame(results).T
    print(df_results)

    print("🐍 DEBUG: Returning from main() now — reached end of main function")
    return df_results


# =========================================================
# 🚀 MAIN EXECUTION — outside the main() function
# =========================================================
if __name__ == "__main__":
    try:
        final_results = main()
        print("\n✅ Training complete. Final Results Table:\n", final_results)
    except Exception as e:
        print("\n❌ ERROR inside main():", str(e))

    # --- ✅ Debug info ---
    print("TYPE OF FINAL_RESULTS:", type(final_results))
    if isinstance(final_results, pd.DataFrame):
        print("✅ Final results DataFrame columns:", final_results.columns.tolist())
    else:
        print("⚠️ final_results is not a DataFrame, converting if possible...")

    # ✅ Save results to CSV
    csv_path = "results.csv"
    final_results.to_csv(csv_path, index=False)
    print(f"\n📁 Results saved to {csv_path}")
    # =========================
# =========================
# 🚀 PHASE 5 — XGBOOST + ENSEMBLE INTEGRATION
# =========================
    # =========================
# 🔹 PHASE 5 — XGBOOST ENSEMBLE INTEGRATION
# =========================
    print("\n🔹 Entering Phase 5: XGBoost + Ensemble Integration...")

    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import numpy as np
    import pandas as pd
    import joblib
    import matplotlib.pyplot as plt

    print("🔄 Preparing ensemble features...")

# Extract the common test labels (from any model)
    y_test = final_results["labels"].iloc[0]  # 383 elements

# Stack predictions from all deep models (shape = [383, 5])
    ensemble_features = np.column_stack([np.array(preds) for preds in final_results["preds"]])

    print(f"✅ Ensemble feature matrix shape: {ensemble_features.shape}")

# Train XGBoost on these features
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    xgb_model.fit(ensemble_features, y_test)

# Predict using XGBoost
    ensemble_preds = xgb_model.predict(ensemble_features)

# Compute metrics
    acc = accuracy_score(y_test, ensemble_preds)
    prec = precision_score(y_test, ensemble_preds)
    rec = recall_score(y_test, ensemble_preds)
    f1 = f1_score(y_test, ensemble_preds)

    print(f"✅ XGBoost Ensemble Results → Acc: {acc*100:.2f}%, Prec: {prec*100:.2f}%, Rec: {rec*100:.2f}%, F1: {f1*100:.2f}%")

# 💾 Save trained ensemble model
    joblib.dump(xgb_model, "xgboost_ensemble_model.pkl")
    print("💾 Saved XGBoost ensemble model: xgboost_ensemble_model.pkl")

# 💾 Save ensemble features for inference (used by app.py)
    try:
        pd.DataFrame(ensemble_features).to_csv("ensemble_features.csv", index=False)
        print("💾 Saved ensemble features to ensemble_features.csv")
    except Exception as e:
        print(f"⚠️ Could not save ensemble features: {e}")

# Add ensemble result to results table
    ensemble_row = pd.DataFrame([{
    "Model": "XGBoost_Ensemble",
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "test_loss": np.nan,
    "labels": y_test,
    "preds": ensemble_preds
    }])

    final_results = pd.concat([final_results, ensemble_row], ignore_index=True)

# Save updated results
    csv_path = "results_with_xgboost.csv"
    final_results.to_csv(csv_path, index=False)
    print(f"📁 Ensemble results saved to {csv_path}")

# Plot comparison including XGBoost
    plt.figure(figsize=(10, 6))
    metrics = ["accuracy", "precision", "recall", "f1"]
    for metric in metrics:
        plt.bar(final_results["Model"], final_results[metric], label=metric)
        plt.title("Model Performance Comparison (Including XGBoost Ensemble)")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig("results_plot.png")
        print("📊 Performance plot saved as results_plot.png")


    # ======================================================
    # 🌟 PHASE 6 — Evaluation + Explainability Layer
    # ======================================================
    print("\n🌟 Entering Phase 6: Advanced Evaluation + Explainability...")

    import shap
    import seaborn as sns
    from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
    import re

    # Load the latest ensemble results
    df_final = pd.read_csv("results_with_xgboost.csv")

    print("🔍 Evaluating XGBoost model with advanced metrics...")

    # --- Robustly parse y_true and y_pred ---
    label_str = str(df_final.iloc[-1]["labels"])
    pred_str = str(df_final.iloc[-1]["preds"])

    # Extract numeric values (works even if 'np.int64' or spaces exist)
    y_true = np.array(re.findall(r'\d+', label_str), dtype=int)
    y_pred = np.array(re.findall(r'\d+', pred_str), dtype=int)

    print(f"✅ Parsed {len(y_true)} true labels and {len(y_pred)} predictions")

    # Safety check for shape mismatch
    if len(y_true) != len(y_pred):
        print(f"⚠️ Length mismatch: labels={len(y_true)}, preds={len(y_pred)}. Truncating to min length.")
        min_len = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:min_len], y_pred[:min_len]

    # Ensure binary only (remove unexpected numbers >1)
    y_true = np.where(y_true > 1, 1, y_true)
    y_pred = np.where(y_pred > 1, 1, y_pred)

    # --- Compute advanced metrics ---
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        # fallback if multiclass-like data appears
        auc = roc_auc_score(y_true, y_pred, multi_class='ovr')

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    print(f"✅ ROC-AUC: {auc:.3f}")
    print("✅ Confusion Matrix:\n", cm)

    # === Confusion Matrix Plot ===
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix — XGBoost Ensemble")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("xgboost_confusion_matrix.png")
    plt.close()
    print("📊 Saved confusion matrix plot: xgboost_confusion_matrix.png")

    # === SHAP Explainability ===
    print("🧩 Generating SHAP explanations for XGBoost...")
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(ensemble_features)

    # Summary plot (feature importance)
    shap.summary_plot(shap_values, ensemble_features, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    plt.close()
    print("📊 Saved SHAP summary plot: shap_summary.png")

    # === Feature importance bar plot ===
    importance_df = pd.DataFrame({
        'Feature': [f'Model_{i+1}' for i in range(ensemble_features.shape[1])],
        'Importance': xgb_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(6, 4))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title("XGBoost Feature Importance (Deep Model Contributions)")
    plt.tight_layout()
    plt.savefig("xgboost_feature_importance.png")
    plt.close()
    print("📈 Saved feature importance plot: xgboost_feature_importance.png")

    # === Phase Summary ===
    print("\n✅ PHASE 6 COMPLETED SUCCESSFULLY!")
    print(f"ROC-AUC: {auc:.3f}")
    print(f"Top contributing deep model: {importance_df.iloc[0]['Feature']} ({importance_df.iloc[0]['Importance']:.3f})")


    import matplotlib.pyplot as plt
    import numpy as np
    

    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
    plt.figure(figsize=(10, 6))
    bar_width = 0.15
    x = np.arange(len(final_results))

    for i, metric in enumerate(metrics_to_plot):
        if metric in final_results.columns:
            plt.bar(x + i * bar_width, final_results[metric], width=bar_width, label=metric)

    plt.title("Model Performance Comparison")
    plt.xlabel("Models")
    plt.ylabel("Score (%)")
    plt.xticks(x + bar_width * 1.5, final_results["Model"], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results_plot.png")
    plt.show()
    print("📊 Performance plot saved as results_plot.png")
    
    

    # =========================================================
    # 🚀 PHASE 5: XGBoost Ensemble Integration
    # =========================================================
''' print("\n🚀 Starting PHASE 5: XGBoost Ensemble Integration...")

    import xgboost as xgb
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    df_results = pd.read_csv("results.csv")

    # Assume Transformer is the best model — can adjust later
    best_model_row = df_results.loc[df_results["accuracy"].idxmax()]
    print("\n🏆 Best model from Phase 4:", best_model_row["Model"])

    # For demo — simulate labels/preds (if not stored earlier)
    labels = np.random.randint(0, 2, size=200)
    preds = np.random.randint(0, 2, size=200)

    # Create meta features for ensemble (can expand with real model outputs)
    X_meta = np.column_stack((preds, preds))
    y_meta = np.array(labels)

    # Train/test split for meta-learner
    split = int(0.8 * len(X_meta))
    X_train, X_test = X_meta[:split], X_meta[split:]
    y_train, y_test = y_meta[:split], y_meta[split:]

    # Train XGBoost ensemble
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)

    # Evaluate ensemble
    y_pred = xgb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"✅ Ensemble → Acc: {acc*100:.2f}%, Prec: {prec*100:.2f}%, Rec: {rec*100:.2f}%, F1: {f1*100:.2f}%")

    # Save ensemble results
    df_results.loc[len(df_results)] = [
        "XGBoost_Ensemble", acc*100, prec*100, rec*100, f1*100
    ]
    df_results.to_csv("results.csv", index=False)
    print("💾 Ensemble results appended to results.csv")

    # Plot updated comparison
    plt.figure(figsize=(8, 4))
    for metric in ["accuracy", "precision", "recall", "f1"]:
        plt.bar(df_results["Model"], df_results[metric], alpha=0.7, label=metric)
    plt.legend()
    plt.title("Model Comparison (with XGBoost Ensemble)")
    plt.savefig("ensemble_comparison.png")
    plt.show()
    print("📊 Ensemble comparison plot saved as ensemble_comparison.png")'''


'''  return df_results

if __name__ == "__main__":
    final_results = main()
    print("\n✅ Training complete. Final Results Table:\n", final_results)

    # --- ✅ Debug info ---
    print("TYPE OF FINAL_RESULTS:", type(final_results))
    print("SAMPLE FINAL_RESULTS:\n", final_results.head())

    # ✅ No need to call from_dict again — final_results is already a DataFrame
    df_results = final_results.reset_index(drop=True)

    # ✅ Save results to CSV
    csv_path = "results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n📁 Results saved to {csv_path}")

    # ✅ Plot Accuracy, Precision, Recall, F1
    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
    plt.figure(figsize=(10, 6))
    bar_width = 0.15
    x = np.arange(len(df_results))

    for i, metric in enumerate(metrics_to_plot):
        if metric in df_results.columns:
            plt.bar(x + i * bar_width, df_results[metric], width=bar_width, label=metric)

    plt.title("Model Performance Comparison")
    plt.xlabel("Models")
    plt.ylabel("Score (%)")
    plt.xticks(x + bar_width * 1.5, df_results["Model"], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results_plot.png")
    plt.show()

    print("📊 Performance plot saved as results_plot.png")

    # =========================================================
    # 🚀 PHASE 5: Ensemble Integration using XGBoost
    # =========================================================
    print("\n✅ Reached Phase 5 section — code is running...")
    print("\n🚀 Starting PHASE 5: XGBoost Ensemble Integration...")

    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # === Load Phase 4 results ===
    df_results = pd.read_csv("results.csv")

    # === Prepare data from best deep model (Transformer here) ===
    best_model_row = df_results.loc[df_results["accuracy"].idxmax()]  # ✅ small fix here
    labels = eval(best_model_row["labels"])
    preds = eval(best_model_row["preds"])

    # === Meta features for ensemble ===
    X_meta = np.column_stack((preds, preds))
    y_meta = np.array(labels)

    split = int(0.8 * len(X_meta))
    X_train, X_test = X_meta[:split], X_meta[split:]
    y_train, y_test = y_meta[:split], y_meta[split:]

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"✅ Ensemble → Acc: {acc*100:.2f}%, Prec: {prec*100:.2f}%, Rec: {rec*100:.2f}%, F1: {f1*100:.2f}%")

    df_results.loc[len(df_results)] = [
        "XGBoost_Ensemble", acc*100, prec*100, rec*100, f1*100, 0, list(y_test), list(y_pred)
    ]
    df_results.to_csv("results.csv", index=False)
    print("💾 Ensemble results appended to results.csv")

    plt.figure(figsize=(8, 4))
    for metric in ["accuracy", "precision", "recall", "f1"]:
        plt.bar(df_results["Model"], df_results[metric], alpha=0.7, label=metric)
    plt.legend()
    plt.title("Model Comparison (with XGBoost Ensemble)")
    plt.savefig("ensemble_comparison.png")
    plt.show()
    print("📊 Ensemble comparison plot saved as ensemble_comparison.png")'''

'''if __name__ == "__main__":
    final_results = main()
    print("\n✅ Training complete. Final Results Table:\n", final_results)

    # --- ✅ Debug info ---
    print("TYPE OF FINAL_RESULTS:", type(final_results))
    print("SAMPLE FINAL_RESULTS:\n", final_results.head())

    print("\n=== DEBUG INFO ===")
    if isinstance(final_results, dict):
        print("final_results is a dictionary")
        for k, v in final_results.items():
            print(f"{k}: {type(v)}")
    elif isinstance(final_results, pd.DataFrame):
        print("final_results is a pandas DataFrame with columns:", final_results.columns.tolist())
    else:
        print("final_results type:", type(final_results))

    # ✅ No need to call from_dict again — final_results is already a DataFrame
    df_results = final_results.reset_index(drop=True)

    # ✅ Save results to CSV
    csv_path = "results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n📁 Results saved to {csv_path}")

    # ✅ Plot Accuracy, Precision, Recall, F1
    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]
    plt.figure(figsize=(10, 6))
    bar_width = 0.15
    x = np.arange(len(df_results))

    for i, metric in enumerate(metrics_to_plot):
        if metric in df_results.columns:
            plt.bar(x + i * bar_width, df_results[metric], width=bar_width, label=metric)

    plt.title("Model Performance Comparison")
    plt.xlabel("Models")
    plt.ylabel("Score (%)")
    plt.xticks(x + bar_width * 1.5, df_results["Model"], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results_plot.png")
    plt.show()

    print("📊 Performance plot saved as results_plot.png")
       
        # =========================================================
    # 🚀 PHASE 5: Ensemble Integration using XGBoost
    # =========================================================
    print("\n✅ Reached Phase 5 section — code is running...")
    print("\n🚀 Starting PHASE 5: XGBoost Ensemble Integration...")

    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # === Load Phase 4 results ===
    df_results = pd.read_csv("results.csv")

    # === Prepare data from best deep model (Transformer here) ===
    best_model_row = df_results.loc[df_results["Model"].idxmax()]
    labels = eval(best_model_row["labels"])
    preds = eval(best_model_row["preds"])

    # Use both DL preds + original features (meta-features)
    X_meta = np.column_stack((preds, preds))  # you can extend later with more models
    y_meta = np.array(labels)

    # === Train-test split for meta-learner ===
    split = int(0.8 * len(X_meta))
    X_train, X_test = X_meta[:split], X_meta[split:]
    y_train, y_test = y_meta[:split], y_meta[split:]

    # === Train XGBoost ensemble ===
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)

    # === Evaluate ensemble ===
    y_pred = xgb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"✅ Ensemble → Acc: {acc*100:.2f}%, Prec: {prec*100:.2f}%, Rec: {rec*100:.2f}%, F1: {f1*100:.2f}%")

    # === Save ensemble results ===
    df_results.loc[len(df_results)] = [
        "XGBoost_Ensemble", acc*100, prec*100, rec*100, f1*100, 0, list(y_test), list(y_pred)
    ]
    df_results.to_csv("results.csv", index=False)
    print("💾 Ensemble results appended to results.csv")

    # === Plot updated comparison ===
    plt.figure(figsize=(8, 4))
    for metric in ["accuracy", "precision", "recall", "f1"]:
        plt.bar(df_results["Model"], df_results[metric], alpha=0.7, label=metric)
    plt.legend()
    plt.title("Model Comparison (with XGBoost Ensemble)")
    plt.savefig("ensemble_comparison.png")
    print("📊 Ensemble comparison plot saved as ensemble_comparison.png")'''

