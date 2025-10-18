#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=.
echo "[1/4] Generating synthetic data..."
python src/data/simulate.py
echo "[2/4] Training tiny DL predictor..."
python src/train_dl.py
echo "[3/4] (Optional) Training RL agent (requires stable-baselines3)..."
python src/rl/train_rl.py || echo "RL training skipped or failed (install stable-baselines3)."
echo "[4/4] Starting edge API on :8000"
uvicorn src.edge.server:app --host 0.0.0.0 --port 8000
