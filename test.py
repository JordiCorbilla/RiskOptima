# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 20:10:17 2025

@author: jordi
"""

import yfinance as yf
import matplotlib.pyplot as plt

# Define the ticker symbol
ticker_symbol = "AAPL"

# Download historical data for the past 1 year
data = yf.download(ticker_symbol, period="1y")

# Display the first few rows of the data
print(data.head())

# Plot the closing prices
plt.figure(figsize=(10, 5))
plt.plot(data.index, data['Close'], label='Close Price')
plt.title(f"{ticker_symbol} Closing Prices - Past 1 Year")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.show()