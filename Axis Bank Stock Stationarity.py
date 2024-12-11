import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt

# Define the start and end dates for data download
start = dt.datetime(2021, 6, 1)
end = dt.datetime(2024, 11, 26)
symbol = 'AXISBANK.NS'

# Download historical stock data using yfinance
stk_data = yf.download(symbol, start=start, end=end)


import pandas as pd
# Create a date range with daily frequency
all_date = pd.date_range(start, end, freq='D')
all_date

dummyDate = stk_data.reindex(all_date).fillna(method='ffill')
dummyDate

import matplotlib.pyplot as plt
plt.plot(stk_data["Close"])

from pandas import read_csv
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(dummyDate["Close"], model='multiplicative')
plt.figure(figsize=(16,5))
result.plot()
plt.show()

from pandas import read_csv
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(dummyDate["Close"], model='additive')
plt.figure(figsize=(16,5))
result.plot()
plt.show()

import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Plot the original data to observe non-stationarity
plt.figure(figsize=(10, 5))
plt.plot(stk_data['Close'], label='Original Close Price', color='blue')
plt.title("Original Close Price (Non-Stationary Data)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()

# Check stationarity using Augmented Dickey-Fuller test
result = adfuller(stk_data['Close'])
print("ADF Statistic:", result[0])
print("p-value:", result[1])
if result[1] > 0.05:
    print("Data is non-stationary. Applying differencing...")

# Apply first differencing
stk_data['Close_diff'] = stk_data['Close'].diff()

# Plot the differenced data
plt.figure(figsize=(10, 5))
plt.plot(stk_data['Close_diff'], label='Differenced Close Price', color='orange')
plt.title("Differenced Close Price (Stationary Data)")
plt.xlabel("Date")
plt.ylabel("Differenced Price")
plt.legend()
plt.show()

# Check stationarity again
result_diff = adfuller(stk_data['Close_diff'].dropna())
print("ADF Statistic after differencing:", result_diff[0])
print("p-value after differencing:", result_diff[1])
if result_diff[1] < 0.05:
    print("Data is now stationary.")


#Plot ACF and PACF to determine ARIMA parameters
plot_acf(stk_data['Close'].dropna(), lags=20)
plot_pacf(stk_data['Close'].dropna(), lags=20)
plt.show()


#Plot ACF and PACF to determine ARIMA parameters
plot_acf(stk_data['Close_diff'].dropna(), lags=20)
plot_pacf(stk_data['Close_diff'].dropna(), lags=20)
plt.show()