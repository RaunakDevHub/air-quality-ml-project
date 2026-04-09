import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

data = pd.read_csv('../data/air_quality.csv')
data = data.dropna()

X = data[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']]
y = data['AQI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred))

def check_alert(aqi):
    if aqi > 300:
        return "Hazardous 🚨"
    elif aqi > 200:
        return "Very Unhealthy ⚠️"
    elif aqi > 100:
        return "Moderate"
    else:
        return "Good"

sample = model.predict([X_test.iloc[0]])
print("Predicted AQI:", sample[0])
print("Alert:", check_alert(sample[0]))

plt.plot(y_test.values[:50], label="Actual")
plt.plot(pred[:50], label="Predicted")
plt.legend()
plt.title("AQI Prediction")
plt.show()
