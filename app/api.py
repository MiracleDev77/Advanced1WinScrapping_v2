from fastapi import FastAPI
from pydantic import BaseModel
from app.predictor import CasinoPredictor

app = FastAPI()
predictor = CasinoPredictor()

class PredictionRequest(BaseModel):
    data: list  # Format brut des tours

@app.post("/predict")
async def predict(request: PredictionRequest):
    return predictor.predict(request.data)

@app.get("/health")
async def health_check():
    return {"status": "OK"}