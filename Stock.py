import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# preprocess the data 
def preprocess_data(stock_data):
    stock_data = stock_data[['Close']]  # We are only using the 'Close' price for simplicity
    stock_data['Prediction'] = stock_data['Close'].shift(-1)  # Next day's closing price (target variable)
    stock_data.dropna(inplace=True)  # Remove rows with NaN values (the last row will have NaN in 'Prediction')
    return stock_data

# Function to train the model and make predictions
def train_and_predict(stock_data):
    # Features (input data): We use the 'Close' price to predict the next day's 'Close' price
    X = stock_data[['Close']]
    # Target variable (output): The next day's 'Close' price
    y = stock_data['Prediction']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    #  Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Squared Error
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    return model, X_test, y_test, y_pred

# visualize results
def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(10,6))
    plt.plot(y_test.index, y_test, color='blue', label='Actual Price')
    plt.plot(y_test.index, y_pred, color='red', label='Predicted Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.show()

# Main function
def main():
    # Define stock ticker and date range
    ticker = 'AAPL'  # Example: Apple Inc.
    start_date = '2019-01-01'
    end_date = '2023-01-01'

    # Fetch the stock data
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    
    # Preprocess the data
    processed_data = preprocess_data(stock_data)

    # Train the model and make predictions
    model, X_test, y_test, y_pred = train_and_predict(processed_data)

    # Visualize the results
    plot_predictions(y_test, y_pred)

if __name__ == "__main__":
    main()
