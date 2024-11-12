import yfinance as yf
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Define the ticker symbol
ticker = input('ticker: ')

# Fetch data for a specific period
data = yf.download(ticker)

# Load the data (ensure you have your DataFrame `data`)
data = data[['Adj Close']]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare training data
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

# Create sequences
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i - seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

# Sequence length of 60 days (adjust as needed)
seq_length = 60
x_train, y_train = create_sequences(train_data, seq_length)
x_test, y_test = create_sequences(test_data, seq_length)

# Reshape data to 3D format for LSTM: (samples, timesteps, features)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))  # Prediction of the next price

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Make predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)  # Reverse scaling

# Display predictions alongside actual values
test_data_actual = scaler.inverse_transform(test_data[seq_length:])
results = pd.DataFrame({"Actual": test_data_actual.flatten(), "Predicted": predictions.flatten()}, index=data.index[-len(test_data_actual):])

# Calculate error metrics
mae = mean_absolute_error(results['Actual'], results['Predicted'])
mse = mean_squared_error(results['Actual'], results['Predicted'])
rmse = np.sqrt(mse)
results['Percentage Error'] = abs((results['Actual'] - results['Predicted']) / results['Actual']) * 100
mape = results['Percentage Error'].mean()

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Save the results to a CSV file named after the ticker
results.to_csv(f'{ticker}_predictions.csv', index=True)

# Reload the CSV for verification and graphing with date index
loaded_results = pd.read_csv(f'{ticker}_predictions.csv', index_col=0, parse_dates=True)

# Plotting the data with dates and error metrics
plt.figure(figsize=(12, 6))
plt.plot(loaded_results.index, loaded_results['Actual'], label='Actual', linestyle='-', marker='o')
plt.plot(loaded_results.index, loaded_results['Predicted'], label='Predicted', linestyle='--', marker='x')

# Format the x-axis for better date display
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=45)  # Rotate dates for better readability

# Adding error metrics text box on the side of the plot
textstr = f"MAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAPE: {mape:.2f}%"
plt.gca().text(1.05, 0.5, textstr, transform=plt.gca().transAxes, 
               fontsize=10, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

# Labels and title
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'{ticker} Actual vs Predicted Stock Prices')
plt.legend()
plt.grid(True)
plt.tight_layout()  # Adjust layout to avoid overlap
plt.savefig(f'{ticker}_predictions_with_errors.jpg')
plt.show()
