from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

MODEL_PATH = "models/model_v1.joblib"

app = FastAPI()

model = joblib.load(MODEL_PATH)


class TaxiRequest(BaseModel):
    trip_distance: float
    passenger_count: int
    tpep_pickup_datetime: str
    PULocationID: int
    DOLocationID: int
    VendorID: int
    RatecodeID: int


@app.get("/")
def home():
    return {"message": "NYC Taxi Fare Prediction API is running"}


@app.post("/predict")
def predict(data: TaxiRequest):
    input_df = pd.DataFrame([data.dict()])

    prediction = model.predict(input_df)[0]

    return {
        "predicted_fare": round(float(prediction), 2)
    }
