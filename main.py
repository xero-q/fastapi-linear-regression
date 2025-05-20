# main.py
from fastapi import FastAPI
from pydantic import BaseModel, conint
from joblib import load
import numpy as np

scaler = load('./scaler.joblib')
model = load('./linear_model.joblib')

app = FastAPI()

# Input schema


class InputData(BaseModel):
    height: conint(ge=50, le=300)
    gender: conint(ge=0, le=1)


class Prediction(BaseModel):
    weight: float


@app.post("/predict", response_model=Prediction)
def predict(data: InputData):
    input_array = np.array([[data.height, data.gender]])
    input_scaled = scaler.transform(input_array)

    # Predict
    prediction = model.predict(input_scaled)
    return {"weight": round(float(prediction[0]), 2)}
