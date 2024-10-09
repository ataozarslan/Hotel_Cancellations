from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from joblib import load

xgb_model = load('hotel_cancellation_model.pkl')

app = FastAPI()

class PredictionInput(BaseModel):
    lead_time: int
    no_of_special_requests: int
    avg_price_per_room: float
    market_segment_type_Online: int
    arrival_month: int
    arrival_date: int
    arrival_year: int
    no_of_weekend_nights: int
    no_of_week_nights: int
    no_of_adults: int

@app.post("/predict")
async def predict(input_data: PredictionInput):
    input_df = pd.DataFrame([input_data.dict()])
    prediction = (xgb_model.predict_proba(input_df)[:,1] > 0.3).astype(int)
    probability = xgb_model.predict_proba(input_df)[0][1] * 100
    return {"prediction": "Will be Canceled" if prediction[0] else "Will Not be Canceled", "probability": probability}