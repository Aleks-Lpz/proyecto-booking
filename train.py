import os
import pandas as pd
import pathlib
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from joblib import dump

# Leer el número de hojas desde la variable de entorno (pasada por GitHub Actions o local)
max_leaves_env = os.environ.get('INPUT_MAX_LEAF_NODES') or os.environ.get('MAX_LEAF_NODES', '50')
max_leaf_nodes = int(max_leaves_env)
print(f"--> Entrenando modelo con max_leaf_nodes = {max_leaf_nodes}")

# 1. Cargar el dataset y limpiar nulos
df = pd.read_csv('hotel_bookings.csv').dropna(subset=['adr'])

features = ['hotel', 'market_segment', 'distribution_channel', 'customer_type']
X = df[features]
y = df['adr']

# 2. Pipeline con transformador
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), features)
    ]
)

modelo_final = Pipeline(steps=[
    ('preprocesador', preprocessor),
    ('modelo', DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1))
])

modelo_final.fit(X, y)

# 3. Guardar el modelo entrenado
pathlib.Path('model').mkdir(parents=True, exist_ok=True)
dump(modelo_final, 'model/hotel-model-v1.joblib')

print(f"¡Modelo entrenado con {max_leaf_nodes} hojas y guardado exitosamente!")
