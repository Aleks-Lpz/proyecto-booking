import pandas as pd
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('hotel_bookings.csv').dropna(axis=0)

X = pd.get_dummies(df[['hotel', 'market_segment', 'distribution_channel', 'customer_type']], drop_first=True)
y = df['adr']

print("Filas y columnas en X:", X.shape)
print("Target (y) muestreo:")
print(y.head())

model = DecisionTreeRegressor(random_state=1)
print("Entrenando modelo...")
model.fit(X, y)
print("¡Entrenamiento completado!")

print("\nPredicciones para los primeros 5 registros:")
print(model.predict(X.head()))
print("Tarifas (adr) reales:")
print(y.head().values)
