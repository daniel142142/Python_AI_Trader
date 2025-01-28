# Import Libraries
import torch
import torch.nn as nn
import numpy
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

ticker = "AAPL"
data = yf.download(ticker, start="2015-01-01", end="2025-01-01")
closing_prices = data['Close'].values

# Normalize Data
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_prices = scaler.fit_transform(closing_prices.reshape(-1, 1))


def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

sequence_length = 30  # Use the last 30 days to predict the next day
X, y = create_sequences(normalized_prices, sequence_length)

# Split into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate= 0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=5, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)


    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden = self.dropout(hidden[-1])
        return self.fc(hidden)

input_size = 1
hidden_size = 50  # Number of LSTM units
model = LSTMModel(input_size, hidden_size)

# optimising the model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the Model
epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()  # Reset gradients
    predictions = model(X_train)  # Forward pass
    loss = criterion(predictions, y_train)  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluate the Model
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    test_loss = criterion(test_predictions, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")

# Creating graph of predictions
actual_prices = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))
predicted_prices = scaler.inverse_transform(test_predictions.numpy().reshape(-1, 1))

# calculating mse
mse = numpy.mean((actual_prices - predicted_prices) ** 2)

plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Prices', color='blue')
plt.plot(predicted_prices, label='Predicted Prices', color='red')
plt.legend()
plt.title("Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()