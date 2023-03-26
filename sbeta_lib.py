import yfinance as yf
import numpy as np

def calculate_beta(stock_data, market: str,start_date) -> float:

    # Calculate the stock's returns
    stock_returns = stock_data["Close"].pct_change().dropna()
    
    # If a market index is specified, get the market data and calculate the market's returns
    if market:
        market_data = yf.Ticker(market).history(start=start_date)
        market_data = market_data.tz_localize(None)
        market_returns = market_data["Close"].pct_change().dropna()
    else:
        market_returns = np.ones_like(stock_returns)
    
    # Calculate the stock's beta using the returns
    beta = stock_returns.cov(market_returns) / market_returns.var()
    
    return beta, market_data

def get_beta(stock_data):
    
    # Use 3 years back (common standard)
    beta, market_data = calculate_beta(stock_data, "^GSPC",start_date = stock_data.index[0] )
    
    return round(beta,4), market_data

if __name__ == "__main__":
    stock_data = yf.Ticker('AAPL').history(period="2y").tz_localize(None)
    beta, market_data = get_beta(stock_data=stock_data)
    print(beta)
