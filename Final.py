import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Function to fetch historical stock data
def fetch_stock_data(symbol, start_date='2010-01-01', end_date='2023-01-01'):
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        if stock_data.empty:
            print(f"Error: No data found for symbol {symbol}. Please check the symbol and try again.")
            return None
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# Function to fetch additional stock info
def fetch_stock_info(symbol):
    try:
        stock = yf.Ticker(symbol)
        stock_info = stock.info
        return stock_info
    except Exception as e:
        print(f"Error fetching info for {symbol}: {e}")
        return {}

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

# Function to plot stock data with RSI, and add resistance/support lines
def plot_stock_data_with_rsi(stock_data, symbol, market_cap):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Close price
    ax1.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create secondary y-axis for RSI
    ax2 = ax1.twinx()
    ax2.plot(stock_data.index, stock_data['RSI'], label='RSI', color='orange', linestyle='--')
    ax2.set_ylabel('RSI', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.axhline(70, color='red', linestyle='--', linewidth=0.8, label='Overbought (70)')
    ax2.axhline(30, color='green', linestyle='--', linewidth=0.8, label='Oversold (30)')

    # Calculate resistance and support levels (based on historical data)
    resistance = stock_data['Close'].max() * 1.05  # Resistance 5% above max closing price
    support = stock_data['Close'].min() * 0.95    # Support 5% below min closing price
    
    # Ensure resistance and support are scalar values (not pandas Series)
    resistance_value = float(resistance)  # Convert to scalar if needed
    support_value = float(support)        # Convert to scalar if needed
    
    # Add resistance and support lines
    ax1.axhline(resistance_value, color='red', linestyle='--', linewidth=1, label=f'Resistance: ${resistance_value:.2f}')
    ax1.axhline(support_value, color='green', linestyle='--', linewidth=1, label=f'Support: ${support_value:.2f}')

    # Display the Market Cap as text on the graph
    ax1.text(0.95, 0.95, f'Market Cap: ${market_cap:.2f}', ha='right', va='top', transform=ax1.transAxes,
             fontsize=12, color='purple', weight='bold', bbox=dict(facecolor='white', edgecolor='purple', boxstyle='round,pad=0.5'))

    # Title and grid
    plt.title(f'{symbol} Stock Price and RSI with Resistance/Support')
    plt.grid(True)

    # Move the legend outside the plot area
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)  # Place the legend outside the plot area
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 0.6), fontsize=10)  # For the RSI and horizontal lines

    plt.tight_layout()  # Adjust layout to fit legend and labels
    plt.show()

# Function to display additional stock information
def display_stock_info(stock_info):
    print("\033[35m\033[1m--- Stock Information ---\033[0m")
    print(f"  \033[1mCompany Name:\033[0m {stock_info.get('longName', 'N/A')}")
    print(f"  \033[1mSymbol:\033[0m {stock_info.get('symbol', 'N/A')}")
    print(f"  \033[1mSector:\033[0m {stock_info.get('sector', 'N/A')}")
    print(f"  \033[1mIndustry:\033[0m {stock_info.get('industry', 'N/A')}")
    print(f"  \033[1mMarket Cap:\033[0m {stock_info.get('marketCap', 'N/A')}")
    print(f"  \033[1mPE Ratio:\033[0m {stock_info.get('trailingPE', 'N/A')}")
    print(f"  \033[1mDividend Yield:\033[0m {stock_info.get('dividendYield', 'N/A')}")
    print(f"  \033[1mBeta:\033[0m {stock_info.get('beta', 'N/A')}")
    print(f"  \033[1mCountry:\033[0m {stock_info.get('country', 'N/A')}")
    print(f"  \033[1mCurrency:\033[0m {stock_info.get('currency', 'N/A')}")
    print(f"  \033[1mPrevious Close:\033[0m \033[32m${stock_info.get('regularMarketPreviousClose', 'N/A')}\033[0m")
    print(f"  \033[1mDay High:\033[0m \033[32m${stock_info.get('dayHigh', 'N/A')}\033[0m")
    print(f"  \033[1mDay Low:\033[0m \033[32m${stock_info.get('dayLow', 'N/A')}\033[0m\n")

# Main function
def main():
    # Print red separator line at the beginning with "⏙" symbol
    print("\033[31m⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙\033[0m")

    # Welcome message
    print("\033[35m\033[1mWelcome to the Stock Market Predictor!\033[0m\n")  # Purple and bold

    # Get user input for stock symbol
    symbol = input("Enter the stock symbol (e.g., AAPL for Apple): ").upper()

    # Fetch stock data and check if it was successfully retrieved
    stock_data = fetch_stock_data(symbol)
    if stock_data is None:
        return

    # Fetch additional stock information
    stock_info = fetch_stock_info(symbol)

    # Display stock information with purple header
    display_stock_info(stock_info)

    # Get market cap
    market_cap = stock_info.get('marketCap', 0)  # Ensure it's a default if not available

    # Calculate RSI
    stock_data = calculate_rsi(stock_data)

    # Display the most recent stock data in bold purple
    print("\033[35m\033[1mMost recent stock data:\033[0m")
    print(stock_data.tail(1).to_string())  # Display the last row of the data (most recent day)
    print("\n" + "="*50 + "\n")

    # Plot the stock data with RSI and resistance/support lines, including Market Cap
    plot_stock_data_with_rsi(stock_data, symbol, market_cap)

    # Calculate current worth and growth
    current_price = stock_data['Close'][-1]
    initial_price = stock_data['Close'][0]
    growth_percentage = ((current_price - initial_price) / initial_price) * 100

    print(f"\033[1mCurrent price of {symbol}:\033[0m \033[32m${current_price:.2f}\033[0m")
    print(f"\033[1mInitial price (start of data):\033[0m \033[32m${initial_price:.2f}\033[0m")
    print(f"\033[1mGrowth over the period:\033[0m \033[34m{growth_percentage:.2f}%\033[0m\n")
    
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
    print(f'\033[1mMean Squared Error:\033[0m \033[31m{mse:.2f}\033[0m')

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
    print(f"\n\033[1mPredicted next day's closing price for {symbol}:\033[0m \033[32m${next_day_prediction[0]:.2f}\033[0m")

    # Print red separator line at the end with "⏙" symbol
    print("\033[31m⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙⏙\033[0m")

if __name__ == "__main__":
    main()
    