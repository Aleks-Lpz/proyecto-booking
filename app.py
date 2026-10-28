from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pathlib
import pandas as pd

app = FastAPI(title="Hotel ADR Prediction API", version="1.0.0")

# Cargar el modelo guardado
model_path = pathlib.Path('model/hotel-model-v1.joblib')
model = load(model_path)

class BookingInput(BaseModel):
    hotel: str
    market_segment: str
    distribution_channel: str
    customer_type: str
    lead_time: float = 40.0              # Valor por defecto
    adults: int = 2                     # Valor por defecto
    total_of_special_requests: int = 0   # Valor por defecto

class BookingOutput(BaseModel):
    predicted_adr: float

@app.post('/predict', response_model=BookingOutput)
def predict_adr(data: BookingInput):
    # Crear un DataFrame con la estructura exacta que espera el pipeline
    input_df = pd.DataFrame([{
        'hotel': data.hotel,
        'market_segment': data.market_segment,
        'distribution_channel': data.distribution_channel,
        'customer_type': data.customer_type,
        'lead_time': data.lead_time,
        'adults': data.adults,
        'total_of_special_requests': data.total_of_special_requests
    }])
    
    prediction = model.predict(input_df)[0]
    return BookingOutput(predicted_adr=float(prediction))
