import requests
import pandas as pd
import json

YOUR_API_KEY = "IOO8TG5PEK3DL7M3"

def get_stock_earnings(ticker_symbol):
    """Gets the latest and estimated future earnings for a stock symbol.

    Args:
        ticker_symbol (str): The stock symbol of the company.

    Returns:
        pandas.DataFrame: A DataFrame containing the latest and estimated future earnings.
    """

    # Get the latest earnings
    url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker_symbol}&apikey={YOUR_API_KEY}"
    response = requests.get(url)
    earnings = response.json()
    
    # Extract the relevant parts
    annual_earnings = earnings['annualEarnings']
    quarterly_earnings = earnings['quarterlyEarnings']

    # Convert lists to DataFrames
    df_annual_earnings = pd.DataFrame(annual_earnings)
    df_quarterly_earnings = pd.DataFrame(quarterly_earnings)

    return df_annual_earnings, df_quarterly_earnings


if __name__=="__main__":

    stock = input("Enter Stock Symbol : ")
    df_annual_earnings , df_quarterly_earnings = get_stock_earnings(stock)

    # Print the DataFrames
    print("Annual Earnings DataFrame:")
    print(df_annual_earnings.head(5)) # change the number of years

    print("\nQuarterly Earnings DataFrame:")
    print(df_quarterly_earnings.head(8)) # change the number of quarters


