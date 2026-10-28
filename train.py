import pandas as pd
import pathlib
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from joblib import dump

# 1. Cargar el dataset y limpiar nulos
df = pd.read_csv('hotel_bookings.csv').dropna(subset=['adr'])

# Definir variables predictoras (X) y objetivo (y)
features = ['hotel', 'market_segment', 'distribution_channel', 'customer_type']
X = df[features]
y = df['adr']

print("Entrenando el modelo de predicción de tarifas de hotel...")

# 2. Pipeline con transformador para procesar texto automáticamente
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), features)
    ]
)

modelo_final = Pipeline(steps=[
    ('preprocesador', preprocessor),
    ('modelo', DecisionTreeRegressor(max_leaf_nodes=50, random_state=1))
])

modelo_final.fit(X, y)

# 3. Guardar el modelo entrenado
pathlib.Path('model').mkdir(parents=True, exist_ok=True)
dump(modelo_final, 'model/hotel-model-v1.joblib')

print("¡Modelo de hotel entrenado y guardado exitosamente en model/hotel-model-v1.joblib!")
