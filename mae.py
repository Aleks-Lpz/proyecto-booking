import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('hotel_bookings.csv').dropna(axis=0)

X = pd.get_dummies(df[['hotel', 'market_segment', 'distribution_channel', 'customer_type']], drop_first=True)
y = df['adr']

model = DecisionTreeRegressor(random_state=1)
model.fit(X, y)

predictions = model.predict(X)
mae_in_sample = mean_absolute_error(y, predictions)
print("MAE In-Sample (Entrenamiento):", mae_in_sample)
