from fastapi import FastAPI
from joblib import load
import numpy as np
from pydantic import BaseModel

app = FastAPI()

@app.get('/')
def home():
    return 'Welcome to Hotel Cancellation Prediction API!'

class MyData(BaseModel):
    lead_time:int
    no_of_special_requests:int
    avg_price_per_room:int
    market_segment_type_Online:int
    arrival_month:int
    arrival_date:int
    arrival_year:int
    no_of_weekend_nights:int
    no_of_week_nights:int
    no_of_adults:int

@app.post('/predict')
def predict(data:MyData):
    model = load('xgb_model.pkl')
    features = np.array([[data.lead_time, data.no_of_special_requests, data.avg_price_per_room,
                          data.market_segment_type_Online, data.arrival_month, data.arrival_date,
                          data.arrival_year, data.no_of_weekend_nights, data.no_of_week_nights, data.no_of_adults]])
    prediction = (model.predict_proba(features)[:,1] > 0.3).astype(int)
    prediction_proba = model.predict_proba(features)[:,1]
    return {"Is Canceled":int(prediction), "Cancelation Probability (%)":float(prediction_proba*100)}

# python -m uvicorn main:app --reload