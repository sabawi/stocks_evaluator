import pandas as pd
import yfinance as yf
import numpy as np

def annual_income_stmt_page(stock, sdata):
    txt_page=f"\n{'=' * 30} Annual Income Statement {'=' * 30} "
    txt_page+=sdata.income_stmt
    
    return txt_page

if __name__== "__main__":
    stock = input("Enter Stock Symbol:").strip().upper()
    
    sdata = yf.Ticker(stock)
    timezone = sdata.earnings_dates.index.tz
    today=  pd.Timestamp('today').tz_localize(timezone)
    print(f"Stock = {stock} on {today}")
    latest_price= round(sdata.history(period="2d",interval='1m').iloc[-1].Close,2)
    # print(f"\n{'=' * 30} Annual Income Statement {'=' * 30} ")
    # print(sdata.income_stmt)
    
    print(annual_income_stmt_page(stock=stock,sdata=sdata))
    
    print(f"\n{'=' * 30} Last Annual EPS {'=' * 30} ")
    latest_report_a = sdata.income_stmt.columns.max()  # Get the latest date
    latest_eps_a = sdata.income_stmt.loc['Basic EPS', latest_report_a]  # Access the value

    print(f"Annual Earning per Share (EPS) on {latest_report_a} = {latest_eps_a}")
    
    
    print(f"\n{'=' * 30} Quarterly Income Statement {'=' * 30} ")
    print(sdata.quarterly_income_stmt)
    
    print(f"\n{'=' * 30} Last Quarterly EPS {'=' * 30} ")
    latest_report_q = sdata.quarterly_income_stmt.columns.max()  # Get the latest date
    latest_eps_q = sdata.quarterly_income_stmt.loc['Basic EPS', latest_report_q]  # Access the value

    print(f"Quarterly Earning per Share (EPS) on {latest_report_q} = {latest_eps_q}")
    
    print(f"\n{'=' * 30} Earnings Dates {'=' * 30} ")
    print(sdata.earnings_dates)
    
    
    future_eps_estimates = sdata.earnings_dates[sdata.earnings_dates.index > today]
    
    eps_estimate = None
    eps_estimate_date = None
    for i in range(len(future_eps_estimates)):
        # print(f"{i}---{future_eps_estimates.index[i]} value {future_eps_estimates.iloc[i]['EPS Estimate']}")
        if pd.notna( future_eps_estimates.iloc[i]['EPS Estimate']):
            eps_estimate = future_eps_estimates.iloc[i]['EPS Estimate']
            eps_estimate_date = future_eps_estimates.index[i]
        
    
    # # Filter for dates with both EPS Estimate and Reported EPS (potential reporting dates)
    # potential_reporting_dates = sdata.earnings_dates[sdata.earnings_dates['EPS Estimate'].notna() & sdata.earnings_dates['Reported EPS'].notna()]

    # # Filter for future reporting dates (excluding today)
    # future_reporting_dates = potential_reporting_dates[potential_reporting_dates.index > today]

    # # Check if there are any future reporting dates
    # if not future_reporting_dates.empty:
    #     # Get EPS Estimate and date for the earliest upcoming reporting date
    #     eps_estimate = future_reporting_dates.iloc[0]['EPS Estimate']
    #     eps_estimate_date = future_reporting_dates.index[0]
    # else:
    #     # Handle no upcoming reporting dates (optional)
    #     eps_estimate = None
    #     eps_estimate_date = None

    # Print the results (focusing on reporting dates)
    print(f"\n{'=' * 30} Upcoming Reporting Date EPS Estimate {'=' * 30} ")
    print(f"Estimated EPS due report on {eps_estimate_date} = {eps_estimate}")

    print(f"\n{'=' * 30} Recommendations Summary {'=' * 30} ")
    print(sdata.recommendations_summary)
    
    print(f"\n{'=' * 30} Upgrades/Downgrades {'=' * 30} ")
    # print(sdata.upgrades_downgrades)
    # from IPython.display import display
    # Set the maximum number of rows and columns to display
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_columns', 10)
    
    print(sdata.upgrades_downgrades[sdata.upgrades_downgrades.index > "2023-06-01"])
    
    print(f"\n{'=' * 30} Valuation of Stock {'=' * 30} ")
    
    print(f"Current price for {stock} = ${round(latest_price,2)}")
    if latest_eps_a > 0.0:
        print(f"Current P/E = {round(latest_price/latest_eps_a,2)} based on last earnings on {latest_report_a} of EPS={latest_eps_a} and current price=${latest_price}")
    else:
        print(f"Current P/E = N/A <Negative>  based on last earnings on {latest_report_a} of EPS={latest_eps_a} and current price =${latest_price}")
        
    if eps_estimate > 0.0:
        print(f"Forward P/E = {round(latest_price/eps_estimate,2)} based on estimates for earnings on {eps_estimate_date} of EPS={eps_estimate} and current price=${latest_price}")
    else:
        print(f"Estimated future EPS is Negative = {eps_estimate}. No Forward P/E")
