import pandas as pd
import datetime as dt
import QuantLib as ql
import numpy as np

# Example data retrieval functions (simplified)
# ...

import yfinance as yf

def get_stock_price(symbol):
    data = yf.Ticker(symbol).history(period="2d",interval='1m')
    return data["Close"].iloc[-1]
    
def get_historical_prices(symbol,start_date, end_date):
    return yf.Ticker(symbol).history(interval='1d',start=start_date,end=end_date)

def get_historical_vix(start_date, end_date):
    return get_historical_prices("^VIX",start_date, end_date)
    
    
def get_risk_free_rate():
    # Ticker symbol for 10-year Treasury bond
    treasury_ticker = "^TNX"  # Yahoo Finance symbol for 10-year Treasury yield

    # Retrieve historical data for the Treasury yield
    # treasury_data = yf.download(treasury_ticker, period="1d")
    
    treasury_data = yf.Ticker(treasury_ticker).history(period="2d",interval='1m')
    
    # Get the most recent yield value
    most_recent_yield = treasury_data["Close"].iloc[-1] / 100.0
    date_time = treasury_data.index[-1]
    
    return most_recent_yield, date_time

def calculate_historical_volatility(historical_prices):
    # Calculate daily returns
    historical_prices['Return'] = np.log(historical_prices['Close'] / historical_prices['Close'].shift(1))
    
    # Calculate annualized historical volatility
    trading_days_per_year = 252  # Assuming 252 trading days in a year
    historical_volatility = historical_prices['Return'].std() * np.sqrt(trading_days_per_year)

    return historical_volatility



# Example Greek calculation functions using QuantLib for American options
def calculate_greeks(stock_price, strike_price, time_to_expiry, implied_volatility, risk_free_rate):
    # Convert Python float values to QuantLib Real values
    ql_stock_price = ql.SimpleQuote(stock_price)
    ql_implied_volatility = ql.SimpleQuote(implied_volatility)
    ql_risk_free_rate = ql.SimpleQuote(risk_free_rate)
    
    # Create QuantLib objects
    option_type = ql.Option.Call
    underlying_handle = ql.QuoteHandle(ql_stock_price)
    volatility_handle = ql.QuoteHandle(ql_implied_volatility)
    interest_rate_handle = ql.QuoteHandle(ql_risk_free_rate)
    
    exercise = ql.AmericanExercise(ql.Date(), ql.Date())  # Modify as needed
    payoff = ql.PlainVanillaPayoff(option_type, strike_price)
    option = ql.VanillaOption(payoff, exercise)

    # Set up QuantLib process
    calculation_date = ql.Date()  # Modify as needed
    day_count = ql.Actual360()
    calendar = ql.NullCalendar()
    
    volatility_curve = ql.BlackConstantVol(calculation_date, calendar, volatility_handle, day_count)
    interest_rate_curve = ql.FlatForward(calculation_date, interest_rate_handle, day_count)
    
    process = ql.BlackScholesProcess(underlying_handle, ql.YieldTermStructureHandle(interest_rate_curve), ql.BlackVolTermStructureHandle(volatility_curve))

    # Calculate Greeks using finite difference engine for American options
    engine = ql.FdBlackScholesVanillaEngine(process)
    option.setPricingEngine(engine)

    delta = option.delta()
    gamma = option.gamma()
    theta = option.theta()
    vega = option.vega()
    rho = option.rho()

    return delta, gamma, theta, vega, rho

# Example usage
if __name__ == "__main__":
    symbol = 'AAPL'
    start_date = "2022-01-03"
    end_date = dt.datetime.now().strftime("%Y-%m-%d")
    # print(end_date)
    # end_date = dt.datetime.now().strftime('YYYY-MM-DD HH:MM:SS')
    
    current_stock_price = get_stock_price(symbol)
    historical_prices = get_historical_prices(symbol, start_date, end_date)
    historical_volatility_value = calculate_historical_volatility(historical_prices)
    # historical_vix = get_historical_vix(start_date, end_date)
    risk_free_rate, date_time = get_risk_free_rate()

    # Extract necessary data from historical data
    stock_prices = historical_prices['Close']
    # implied_volatility = historical_vix['Close']

    # Calculate Greeks for a specific option
    strike_price = 200.0
    time_to_expiry = 0.5

    print("Implied Vol :",historical_volatility_value)
    print("risk_free_rate :",risk_free_rate)
    print("time to expiry : ",time_to_expiry)
    delta, gamma, theta, vega, rho = calculate_greeks(current_stock_price, strike_price, time_to_expiry, historical_volatility_value, risk_free_rate)

    print("Delta:", delta)
    print("Gamma:", gamma)
    print("Theta:", theta)
    print("Vega:", vega)
    print("Rho:", rho)
