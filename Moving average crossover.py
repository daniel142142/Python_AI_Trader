import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np

# Download historical data for the S&P 500
ticker = "^GSPC"
data = yf.download(ticker, start="2010-01-01", end="2023-12-31")

# Calculate moving averages
data['SMA50'] = data['Close'].rolling(window=50).mean()
data['SMA200'] = data['Close'].rolling(window=200).mean()

# Generate buy/sell signals
data['Signal'] = 0.0
data['Signal'][50:] = np.where(data['SMA50'][50:] > data['SMA200'][50:], 1.0, 0.0)
data['Position'] = data['Signal'].diff()

# Plot the data
plt.figure(figsize=(14,7))
plt.plot(data['Close'], label='S&P 500', color='blue')
plt.plot(data['SMA50'], label='50-Day SMA', color='green')
plt.plot(data['SMA200'], label='200-Day SMA', color='red')

# Plot buy signals
plt.plot(data[data['Position'] == 1].index,
         data['SMA50'][data['Position'] == 1],
         '^', markersize=10, color='g', lw=0, label='Buy Signal')

# Plot sell signals
plt.plot(data[data['Position'] == -1].index,
         data['SMA50'][data['Position'] == -1],
         'v', markersize=10, color='r', lw=0, label='Sell Signal')

plt.title('S&P 500 - Moving Average Crossover Strategy')
plt.legend(loc='best')
plt.show()

# Backtest the strategy
initial_capital = 100000.0
positions = pd.DataFrame(index=data.index).fillna(0.0)
positions[ticker] = data['Signal']
portfolio = positions.multiply(data['Adj Close'], axis=0)
pos_diff = positions.diff()

portfolio['holdings'] = (positions.multiply(data['Adj Close'], axis=0)).sum(axis=1)
portfolio['cash'] = initial_capital - (pos_diff.multiply(data['Adj Close'], axis=0)).sum(axis=1).cumsum()
portfolio['total'] = portfolio['cash'] + portfolio['holdings']
portfolio['returns'] = portfolio['total'].pct_change()

# Print final portfolio value
print(f"Final Portfolio Value: ${portfolio['total'][-1]:.2f}")

# Plot the portfolio value
plt.figure(figsize=(10, 5))
plt.plot(portfolio['total'], label='Portfolio Value')
plt.title('Portfolio Value Over Time')
plt.legend(loc='best')
plt.show()

# Calculate buy and hold returns
buy_and_hold_returns = (data['Close'][-1] / data['Close'][0]) * initial_capital
print(f"Buy and Hold Strategy Returns: ${buy_and_hold_returns:.2f}")
