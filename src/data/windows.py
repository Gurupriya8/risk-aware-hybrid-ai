# src/data/windows.py
import numpy as np
import pandas as pd

def create_windows(df, feature_cols, seq_len=12, target_col="risk_score"):
    """
    df: DataFrame with columns ['device_id','timestamp', ...features..., target]
    feature_cols: list of feature column names in time order e.g. ['lat','lon','temp','vibration','load']
    seq_len: number of timesteps per window
    returns: X_windows (N, C, T), y_targets (N,)
    """
    X_list, y_list = [], []
    # sort by device and timestamp
    df = df.sort_values(['device_id','timestamp'])
    for device, g in df.groupby('device_id'):
        arr = g[feature_cols].values  # shape (L, C)
        targets = g[target_col].values
        L = len(arr)
        for i in range(L - seq_len + 1):
            window = arr[i:i+seq_len].T  # shape (C, seq_len)
            # choose label at window end (you can choose alternative)
            label = targets[i + seq_len - 1]
            X_list.append(window)
            y_list.append(label)
    if len(X_list) == 0:
        return np.empty((0, len(feature_cols), seq_len)), np.empty((0,))
    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    return X, y
