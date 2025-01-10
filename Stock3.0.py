import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Function to fetch historical stock data
def fetch_stock_data(symbol, start_date='2010-01-01', end_date='2023-01-01'):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0).flatten()  # Ensure it's 1D
    loss = np.where(delta < 0, -delta, 0).flatten()  # Ensure it's 1D

    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi
    return data

# Function to prepare the dataset (use past days to predict future price)
def prepare_data(stock_data, lookback_days=5):
    stock_data['Prediction'] = stock_data['Close'].shift(-lookback_days)
    stock_data = stock_data.dropna()
    X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    y = stock_data['Prediction'].values
    return X, y

# Function to plot stock data with RSI
def plot_stock_data_with_rsi(stock_data, symbol):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Close price
    ax1.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')

    # Create secondary y-axis for RSI
    ax2 = ax1.twinx()
    ax2.plot(stock_data.index, stock_data['RSI'], label='RSI', color='orange', linestyle='--')
    ax2.set_ylabel('RSI', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.axhline(70, color='red', linestyle='--', linewidth=0.8, label='Overbought (70)')
    ax2.axhline(30, color='green', linestyle='--', linewidth=0.8, label='Oversold (30)')
    ax2.legend(loc='upper right')

    plt.title(f'{symbol} Stock Price and RSI')
    plt.grid(True)
    plt.show()

# Main function
def main():
    print("Welcome to the Stock Market Predictor!")
    symbol = input("Enter the stock symbol (e.g., AAPL for Apple): ").upper()
    stock_data = fetch_stock_data(symbol)

    # Calculate RSI
    stock_data = calculate_rsi(stock_data)

    # Display the most recent stock data (latest row)
    print("\nMost recent stock data:")
    print(stock_data.tail(1))  # Display the last row of the data (most recent day)

    # Plot the stock data with RSI
    plot_stock_data_with_rsi(stock_data, symbol)

    # Calculate current worth and growth
    current_price = stock_data['Close'][-1]
    initial_price = stock_data['Close'][0]
    growth_percentage = ((current_price - initial_price) / initial_price) * 100

    print(f"\nCurrent price of {symbol}: ${current_price:.2f}")
    print(f"Initial price (start of data): ${initial_price:.2f}")
    print(f"Growth over the period: {growth_percentage:.2f}%")

    # Prepare the dataset
    X, y = prepare_data(stock_data)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Create and train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print(f'\nMean Squared Error: {mse:.2f}')

    # Display prediction vs actual prices
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual Prices', color='blue')
    plt.plot(predictions, label='Predicted Prices', color='red')
    plt.title(f'{symbol} Prediction vs Actual Prices')
    plt.xlabel('Days')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Predict the next day's price
    last_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-1].values.reshape(1, -1)
    next_day_prediction = model.predict(last_data)
    print(f"\nPredicted next day's closing price for {symbol}: ${next_day_prediction[0]:.2f}")

if __name__ == "__main__":
    main()