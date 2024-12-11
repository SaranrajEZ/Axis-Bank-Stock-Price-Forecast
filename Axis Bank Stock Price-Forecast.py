import pandas as pd
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt

# Define the start and end dates for data download
start = dt.datetime(2021, 6, 1)
end = dt.datetime(2024, 11, 26)
symbol = 'AXISBANK.NS'

# Download historical stock data using yfinance
stk_data = yf.download(symbol, start=start, end=end)
stk_data = stk_data[['Open', 'High', 'Low', 'Close']]

# Normalize the 'Close' column using MinMaxScaler
column = 'Close'
ms = MinMaxScaler()
data1 = ms.fit_transform(stk_data[[column]])  
print('len:', data1.shape)

# Split the data into training and test sets (80% training, 20% test)
training_size = round(len(data1) * 0.80)
X_train = data1[:training_size]
X_test = data1[training_size:]
y_train = data1[:training_size]
y_test = data1[training_size:]

print("X_train length:", X_train.shape)
print("X_test length:", X_test.shape)
print("y_train length:", y_train.shape)
print("y_test length:", y_test.shape)

import warnings
warnings.filterwarnings("ignore")

# Test different ARIMA orders to identify the best performing one
orders = [(1, 1, 1), (1, 1, 2), (2, 3, 1), (2, 2, 2)]

# Loop through the different ARIMA model orders
for order in orders:
    model = ARIMA(data1, order=order)
    model_fit = model.fit()  # Fit the ARIMA model
    
    # Make predictions
    y_pred = model_fit.predict(0, len(data1)-1)
    
    # Evaluate the model using RMSE and MAPE
    rmse = mean_squared_error(data1, y_pred, squared=False)
    mape = mean_absolute_percentage_error(data1, y_pred)
    print(f"Order {order}: RMSE = {rmse}, MAPE = {mape}")

best_order = (1, 1, 1)
model = ARIMA(data1, order=best_order)
model_fit = model.fit()

# Predict using the ARIMA model
y_pred = model_fit.predict(0, len(data1) - 1)

# Calculate evaluation metrics for model performance
rmse = mean_squared_error(data1, y_pred, squared=False)
mape = mean_absolute_percentage_error(data1, y_pred)
r2 = r2_score(data1, y_pred)
print(f"Best Order ({best_order}):")
print(f"RMSE-Testset: {rmse}")
print(f"MAE-Testset: {mape}")
print(f"R^2-Testset: {r2}")
print('*************')

# Plot the actual vs predicted values for normalized data
plt.figure(figsize=(10, 5))
plt.plot(data1, color='blue', label="Actual")
plt.plot(y_pred, color='green', label="Predicted")
plt.title("AXISBANK-Close-AR-Norm")
plt.xlabel("Days")
plt.ylabel("Prices")
plt.legend()
plt.show()

# Inverse transform the normalized data to original scale for better interpretability
aTestNormTable = pd.DataFrame(data1, columns=[column])
actual_stock_price_test_ori = ms.inverse_transform(aTestNormTable)
actual_stock_price_test_oriA = pd.DataFrame(actual_stock_price_test_ori, columns=[column])

# Inverse transform the predicted values back to the original scale
pTestNormTable = pd.DataFrame(y_pred, columns=[column])
predicted_stock_price_test_ori = ms.inverse_transform(pTestNormTable)
predicted_stock_price_test_oriP = pd.DataFrame(predicted_stock_price_test_ori, columns=[column])

# Plot actual vs predicted values for original scale data
plt.figure(figsize=(10, 5))
plt.plot(actual_stock_price_test_oriA, color='blue', label="Actual")
plt.plot(predicted_stock_price_test_oriP, color='green', label="Predicted")
plt.title("AXISBANK-Close-AR-Ori")
plt.xlabel("Days")
plt.ylabel("Prices")
plt.legend()
plt.show()

# Calculate RMSE, MAPE, and R² for the original scale data
rmse_ori = mean_squared_error(actual_stock_price_test_oriA, predicted_stock_price_test_oriP, squared=False)
mape_ori = mean_absolute_percentage_error(actual_stock_price_test_oriA, predicted_stock_price_test_oriP)
r2_ori = r2_score(actual_stock_price_test_oriA, predicted_stock_price_test_oriP)

print("RMSE-Testset (Original):", rmse_ori)
print("MAPE-Testset (Original):", mape_ori)
print("R²-Testset (Original):", r2_ori)

# Use the model to forecast the next 3 days
forecast = model_fit.predict(len(data1), len(data1) + 3)
fTestNormTable = pd.DataFrame(forecast, columns=["Closefore"])
forecast_stock_price_test_ori = ms.inverse_transform(fTestNormTable)
forecast_stock_price_test_oriF = pd.DataFrame(forecast_stock_price_test_ori, columns=["Closefore"])


