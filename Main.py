import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import yfinance as yf

# Define the stock symbol and date range
stock_symbol = "AAPL"
start_date = "2010-01-01"
end_date = "2024-01-23"

# Download historical stock data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Create features (e.g., closing prices) and target variable
features = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
target = stock_data['Close'].shift(-1)  # Shifted by 1 day to predict the next day's closing price

# Drop the last row to align features and target
features = features[:-1]
target = target[:-1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=2, random_state=42)

# Train a RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Use the model to predict future stock prices
future_features = features[-1:].values  # Use the last row of historical data as features
future_prediction = model.predict(future_features)
print(f'Predicted future stock price: {future_prediction[0]}')