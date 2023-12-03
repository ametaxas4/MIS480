import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

def load_data(C:\Users\alexandra\OneDrive\MIS480):
    return pd.read_csv(Sales Orders.csv, parse_dates=['01/01/2023'], index_col='01/01/2023')

def plot_time_series(data, Sales Orders):
    plt.figure(figsize=(10, 6))
    plt.plot(data[Sales Orders])
    plt.title(f'Time Series Plot of Sales Order Data')
    plt.xlabel('01/01/2023')
    plt.ylabel(Sales Orders)
    plt.show()

def test_stationarity(timeseries):
    result = adfuller(timeseries, autolag='AIC')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    if result[1] <= 0.05:
        print("Reject the null hypothesis, data is stationary.")
    else:
        print("Fail to reject the null hypothesis, data is non-stationary.")

def plot_acf_pacf(timeseries, lags=30):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(timeseries, lags=lags, ax=ax1)
    plot_pacf(timeseries, lags=lags, ax=ax2)
    plt.show()
    
