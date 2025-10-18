# src/extract_embeddings.py
import os, numpy as np, torch, pandas as pd
from sklearn.model_selection import train_test_split
from src.models.multimodal import SimpleMultimodalRiskModel
from src.data.windows import create_windows

def run():
    seq_len = 12
    feature_cols = ['lat','lon','temp','vibration','load']
    csv_path = "data/processed/iot_processed.csv"
    model_path = "models/lstm_embedding_model.pth"
    out_dir = "outputs"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(csv_path)
    df['risk_score'] = (df['risk_score'] > 0.5).astype(int)

    X, y = create_windows(df, feature_cols, seq_len=seq_len, target_col='risk_score')
    idx_train, idx_test = train_test_split(range(len(y)), test_size=0.2, stratify=y, random_state=42)
    X_train, y_train = X[idx_train], y[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]

    model = SimpleMultimodalRiskModel(sensor_in=len(feature_cols), d_model=64, exog_in=0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    def get_embs(X_np):
        embs = []
        batch = 64
        with torch.no_grad():
            for i in range(0, len(X_np), batch):
                batch_x = torch.tensor(X_np[i:i+batch], dtype=torch.float32).to(device)
                _, emb = model(batch_x, None)
                embs.append(emb.cpu().numpy())
        return np.vstack(embs)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "train_embeddings.npy"), get_embs(X_train))
    np.save(os.path.join(out_dir, "train_labels.npy"), y_train)
    np.save(os.path.join(out_dir, "test_embeddings.npy"), get_embs(X_test))
    np.save(os.path.join(out_dir, "test_labels.npy"), y_test)
    print("Saved embeddings to", out_dir)

if __name__ == "__main__":
    run()
