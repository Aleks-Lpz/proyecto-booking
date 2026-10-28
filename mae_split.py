import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('hotel_bookings.csv').dropna(axis=0)

X = pd.get_dummies(df[['hotel', 'market_segment', 'distribution_channel', 'customer_type']], drop_first=True)
y = df['adr']

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

model = DecisionTreeRegressor(random_state=1)
model.fit(train_X, train_y)

val_predictions = model.predict(val_X)
mae_out_sample = mean_absolute_error(val_y, val_predictions)
print("MAE Out-of-Sample (Validacion random_state=0):", mae_out_sample)

train_X2, val_X2, train_y2, val_y2 = train_test_split(X, y, test_size=0.2, random_state=0)
model2 = DecisionTreeRegressor(random_state=1)
model2.fit(train_X2, train_y2)

val_predictions2 = model2.predict(val_X2)
mae_test_size = mean_absolute_error(val_y2, val_predictions2)
print("MAE con test_size=0.2:", mae_test_size)
