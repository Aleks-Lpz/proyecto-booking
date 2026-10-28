from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pathlib
import pandas as pd

app = FastAPI(title="Hotel Booking ADR Prediction API", version="1.0.0")

# Cargar el modelo guardado
model_path = pathlib.Path('model/hotel-model-v1.joblib')
model = load(model_path)

# Estructura de entrada limpia basada en las columnas del hotel
class BookingInput(BaseModel):
    hotel: str                    # Ej: "Resort Hotel" o "City Hotel"
    market_segment: str           # Ej: "Online TA", "Direct", etc.
    distribution_channel: str     # Ej: "TA/TO", "Direct", etc.
    customer_type: str            # Ej: "Transient", "Couple", etc.

class BookingOutput(BaseModel):
    predicted_adr: float

@app.post('/predict', response_model=BookingOutput)
def predict_adr(data: BookingInput):
    # Convertir la entrada JSON a un DataFrame de una sola fila
    input_df = pd.DataFrame([data.dict()])
    
    # Predecir utilizando el pipeline entrenado
    prediction = model.predict(input_df)[0]
    
    return BookingOutput(predicted_adr=float(prediction))
