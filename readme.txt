"""
# Stock Price Prediction using LSTM

This script provides an implementation of an LSTM neural network for stock price prediction.
It uses historical stock data to forecast future prices by training on a sequence of past price values.

## Methodology

1. **Data Scaling**: Scales data to a range of 0-1 for optimal neural network performance.
2. **Sequence Creation**: Transforms the data into sequences of a defined length (e.g., 60 days) to create training samples.
3. **Model Architecture**: Builds an LSTM model with:
    - Two LSTM layers with 50 units each
    - Dense output layer to predict the next price
4. **Training**: Trains the model on 80% of the data, validating it on the remaining 20%.
5. **Prediction**: Generates predictions on the test set, reversing scaling to compare with actual prices.

## Evaluation Metrics

The following metrics are used to evaluate model accuracy:

- **Mean Absolute Error (MAE)**: Measures average absolute errors in prediction.
- **Mean Squared Error (MSE)**: Measures average squared errors, penalizing larger errors.
- **Root Mean Squared Error (RMSE)**: Square root of MSE, bringing error units in line with prices.
- **Mean Absolute Percentage Error (MAPE)**: Measures average percentage error, providing insight into prediction accuracy in percentage terms.

"""
