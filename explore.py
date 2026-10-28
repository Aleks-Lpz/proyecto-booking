import pandas as pd

df = pd.read_csv('hotel_bookings.csv').dropna(axis=0)

print("--- Primeras Filas ---")
print(df.head())

print("\n--- Resumen Estadistico de Categóricas ---")
print(df.describe(include='object'))

print("\n--- Columnas del Dataset ---")
print(df.columns)
