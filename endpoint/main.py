import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load

xgb_model = load('hotel_cancellation_model.pkl')

app = FastAPI()

@app.get('/')
def home():
    return 'Welcome to Hotel Cancellation Prediction API!'

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

    inputs = [[
        input_data.lead_time,
        input_data.no_of_special_requests,
        input_data.avg_price_per_room,
        input_data.market_segment_type_Online,
        input_data.arrival_month,
        input_data.arrival_date,
        input_data.arrival_year,
        input_data.no_of_weekend_nights,
        input_data.no_of_week_nights,
        input_data.no_of_adults
    ]]

    probabilities = xgb_model.predict_proba(inputs)
    prediction = np.where(probabilities[:,1] >= 0.3, 'Will be Canceled!', 'Will Not be Canceled!')[0]
    probability = probabilities[0, 1]
    return {"Is Canceled": prediction, "Cancelation Probability (%)": float(probability)}

# python -m uvicorn main:app --reload