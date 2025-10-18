from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch

app = FastAPI(title="Risk-Aware Hybrid AI — Edge API")

class PredictRequest(BaseModel):
    sensor_seq: list  # shape: [B, C, T]
    exogenous: list   # shape: [B, E]

@app.get('/health')
async def health():
    return {'status': 'ok'}

@app.post('/predict')
async def predict(req: PredictRequest):
    # In production: load ONNX/TorchScript; here we return a mock risk score per batch item
    B = len(req.sensor_seq)
    risk = np.random.rand(B).tolist()
    return {'risk_score': risk}
