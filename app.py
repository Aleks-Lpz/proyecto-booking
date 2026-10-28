from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load
import pathlib
import pandas as pd

app = FastAPI(title="Hotel Booking ADR Prediction API", version="1.0.0")

# Configurar CORS para permitir peticiones desde cualquier origen (frontend local o web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo guardado
model_path = pathlib.Path('model/hotel-model-v1.joblib')
model = load(model_path)

class BookingInput(BaseModel):
    hotel: str
    market_segment: str
    distribution_channel: str
    customer_type: str

class BookingOutput(BaseModel):
    predicted_adr: float

@app.post('/predict', response_model=BookingOutput)
def predict_adr(data: BookingInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    return BookingOutput(predicted_adr=float(prediction))
