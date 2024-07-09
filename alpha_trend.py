import yfinance as yf
import pandas as pd

def calculate_alpha_trend(data, coeff=1, AP=14, showsignalsk=True, novolumedata=False):
    # Calculate ATR
    data['TR'] = data['High'] - data['Low']
    data['ATR'] = data['TR'].rolling(window=AP).mean()

    # Calculate thresholds
    data['upT'] = data['Low'] - data['ATR'] * coeff
    data['downT'] = data['High'] + data['ATR'] * coeff

    # Print intermediate values for debugging
    print("Data after calculating TR and ATR:")
    print(data[['TR', 'ATR', 'upT', 'downT']].tail(10))

    # Initialize AlphaTrend column
    data['AlphaTrend'] = 0.0

    for i in range(1, len(data)):
        prev_AlphaTrend = data['AlphaTrend'].iloc[i-1]
        if novolumedata:
            condition = (data['Close'].rolling(window=AP).mean().iloc[i] >= 50)
        else:
            hlc3 = (data['High'] + data['Low'] + data['Close']) / 3
            mfi = 100 * hlc3.rolling(window=AP).mean() / (hlc3.rolling(window=AP).mean().sum())
            condition = (mfi.iloc[i] >= 50)
            # print(f"hlc3={hlc3},mfi={mfi},condition={condition}")

        if condition:
            data.loc[i, 'AlphaTrend'] = max(prev_AlphaTrend, data['upT'].iloc[i])
        else:
            data.loc[i, 'AlphaTrend'] = min(prev_AlphaTrend, data['downT'].iloc[i])

    # Generate Buy and Sell signals
    data['Buy'] = (data['AlphaTrend'] > data['AlphaTrend'].shift(2)) & (data['AlphaTrend'].shift(1) <= data['AlphaTrend'].shift(3))
    data['Sell'] = (data['AlphaTrend'] < data['AlphaTrend'].shift(2)) & (data['AlphaTrend'].shift(1) >= data['AlphaTrend'].shift(3))

    # Add Predicted column
    data['Predicted'] = 0
    data.loc[data['Buy'], 'Predicted'] = 1
    data.loc[data['Sell'], 'Predicted'] = 0

    print(data)
    return data

# Fetching the stock data from Yahoo Finance
stock_symbol = 'AAPL'
data = yf.download(stock_symbol,period='1d', start='2020-01-01', end='2023-01-01')

# Ensure data is properly fetched
if data.empty:
    raise Exception("Failed to fetch data. Please check the stock symbol and date range.")

# Print data after fetching
print("Data after fetching from Yahoo Finance:")
print(data.head(10))
print(data.tail(10))

# Calculate AlphaTrend and signals
result = calculate_alpha_trend(data)

# Filter relevant columns for the final DataFrame
final_result = result[['Close', 'AlphaTrend', 'Buy', 'Sell', 'Predicted']]

# Print the final result
print("Final result:")
print(result.tail(10))
