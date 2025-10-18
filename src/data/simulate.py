"""Synthetic data + black-swan event simulator."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random, os

def generate_iot_stream(num_devices=20, hours=8, freq_minutes=5, seed=42):
    rng = np.random.default_rng(seed)
    records = []
    start = datetime.utcnow().replace(microsecond=0)
    steps = int(hours*60/freq_minutes)
    for d in range(num_devices):
        device_id = f"veh_{d:03d}"
        temp = 2 + 20 * rng.random()
        lat0, lon0 = 12.9 + 0.1*rng.standard_normal(), 77.5 + 0.1*rng.standard_normal()
        for t in range(steps):
            ts = start + timedelta(minutes=t*freq_minutes)
            gps_lat = lat0 + 0.01*rng.standard_normal()
            gps_lon = lon0 + 0.01*rng.standard_normal()
            temp += 0.05*rng.standard_normal()
            vibration = max(0.0, rng.normal(0.5, 0.3))
            load = rng.random()
            records.append({
                "device_id": device_id,
                "timestamp": ts.isoformat(),
                "lat": gps_lat,
                "lon": gps_lon,
                "temp": float(temp),
                "vibration": float(vibration),
                "load": float(load),
            })
    return pd.DataFrame(records)

def write_parquet(df, path="data/synthetic/iot_stream.parquet"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        # Fallback to CSV if pyarrow not installed
        csv_path = path.replace(".parquet", ".csv")
        df.to_csv(csv_path, index=False)
        path = csv_path
    return path

if __name__ == '__main__':
    df = generate_iot_stream()
    out = write_parquet(df)
    print('Wrote synthetic IoT stream to', out, 'rows:', len(df))
