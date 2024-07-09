import yfinance as yf

def get_risk_free_rate():
    # Ticker symbol for 10-year Treasury bond
    treasury_ticker = "^TNX"  # Yahoo Finance symbol for 10-year Treasury yield

    # Retrieve historical data for the Treasury yield
    # treasury_data = yf.download(treasury_ticker, period="1d")
    
    treasury_data = yf.Ticker(treasury_ticker).history(period="5d",interval='1m')
    
    # Get the most recent yield value
    most_recent_yield = treasury_data["Close"].iloc[-1] / 100.0
    date_time = treasury_data.index[-1]
    
    return most_recent_yield, date_time

# Example usage
if __name__ == "__main__":
    risk_free_rate, date_time = get_risk_free_rate()
    print(f"Risk-Free Rate: {round(risk_free_rate,7)} as of {date_time}")
