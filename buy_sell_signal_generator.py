import sqlite3
from datetime import date, datetime
import pandas as pd
from tabulate import tabulate
from Backtester import SignalsBacktester as bt
import yfinance as yf

transact_record = []

def add_data(data, date, symbols):
  """
  Adds a new entry to the provided data dictionary with the date and stock symbols set.
  If the date already exists, it appends the new symbols to the existing set.

  Args:
      data: A dictionary to store date and stock symbol sets.
      date: The date as a string in YYYY-MM-DD format.
      symbols: A list of stock symbols.
  """
  if date not in data:
    data[date] = {symbols}
  else:
    data[date].update({symbols})  # Update with new symbols using set(symbols)


def make_buys_df(data):
    """
  Converts the provided data dictionary to a Pandas DataFrame with columns 'Date' and 'Stock'.

  Args:
      data: A dictionary containing date and stock symbol sets.

  Returns:
      A Pandas DataFrame.
  """
  
    if not data:
        df = pd.DataFrame(columns=['Date', 'Stock'])  # Return empty DataFrame
        df = df.set_index('Date')
    else:
        # print(data)
        df = pd.DataFrame(data)
        
    return df

# Example usage
# data = {}
# add_data(data, "2024-04-12", ["GOOG", "IBM"])
# add_data(data, "2024-04-15", ["AAPL", "MSFT"])
# add_data(data.copy(), "2024-04-12", ["AAPL"])  # Avoid modifying original data

# Create the DataFrame with a copy to prevent unintended modification
# df = create_dataframe(data.copy())

# print(df)

class TradeRecord:
  def __init__(self, transact, date, stock, note):
    self.transact = transact
    self.date = date
    self.stock = stock
    self.note = note

    # Validation logic can be added here
    if self.transact not in ('Buy', 'Sell', 'None'):
        raise ValueError(f"Invalid transaction type: {self.transact}")
    def __str__(self):
        # Return a string representation of the record data
        return f"Transact: {self.transact}, Date: {self.date}, Stock: {self.stock}, Note: {self.note}"

def transact(action,date,stock,note):
    # Create a TradeRecord object
    record = TradeRecord(action, date, stock, note)

    # Append the record to the data list
    transact_record.append(record)
    
def connect_db(db_file):
  """Connects to the database file and returns the connection object.

  Args:
      db_file (str): Path to the database file.

  Returns:
      sqlite3.Connection: The connection object to the database.
  """
  conn = sqlite3.connect(db_file)
  return conn

def query_data(conn, sql_statement):
  """Executes the provided SQL statement and returns the cursor object.

  Args:
      conn (sqlite3.Connection): The connection object to the database.
      sql_statement (str): The SQL statement to be executed.

  Returns:
      sqlite3.Cursor: The cursor object containing the results of the query.
  """
  cursor = conn.cursor()
  cursor.execute(sql_statement)
  return cursor

def process_data(cursor):
    """Processes the data from the cursor and prints buy/sell alerts.

    This function iterates through the cursor data (assumed to be stock information with 
    "Last Run", "Stock", and "Last Price" columns) and simulates a trading process based on 
    specific conditions. It keeps track of a current portfolio and prints buy/sell alerts 
    as dates change and stock presence/absence changes.

    Args:
        cursor (sqlite3.Cursor): The cursor object containing the stock data.
    """
    current_portfolio = {}  # Use a dictionary to store stock symbols with last_run dates
    today_portfolio = {}
    sell_portfolio = {}
    
    processing_date = datetime.strptime("2000-01-01", "%Y-%m-%d")
    for row in cursor.fetchall():
        last_run, stock, last_price = row
        date_last_run =  datetime.strptime(last_run, "%Y-%m-%d")
        stock = stock.strip()
        # Check if it's a new date
        if date_last_run > processing_date: 

            # If there are stocks to sell from the last pass, sell them now
            for sell_stock in sell_portfolio:
                print(f"SELL: {sell_stock} on {processing_date}")
                    
                # Create a TradeRecord object
                transact("Sell", processing_date, sell_stock, "Sell at the next open")          
            
            current_portfolio = today_portfolio.copy()
            today_portfolio = {}
            
            # At this point, current_portfolio comes from the previous date's list (processing_date)
            print("************************************************************")
            print(f"Portfolio for ({processing_date}) Should be : {current_portfolio}")   
            print("************************************************************")


            # update the processing_date to the one in the last record
            processing_date = date_last_run
            print(f"New Date: {processing_date}")
            sell_portfolio = {}
            today_portfolio = {}            
            
            # Check if the stock in the record is in the current portfolio
            # If not transact a Buy and add it to today_portfolio
            if stock not in current_portfolio:                 
                print(f"BUY: {stock} on {last_run} at ${last_price}")         
                
                # Create a TradeRecord object
                transact("Buy", processing_date, stock, "Buy at the next open")   
                
            today_portfolio[stock] = last_run
        # Else the new record is still in the same date as the last pass
        else:
            # if the new record has a stock not in the current portfolio, buy it
            if stock not in set(current_portfolio):
                print(f"BUY: {stock} on {last_run} at ${last_price}")                          
                # Create a TradeRecord object
                
                transact("Buy", processing_date, stock, "Buy at the next open")
                
            today_portfolio[stock] = last_run
            
        sell_portfolio = set(current_portfolio) - set(today_portfolio)

    # If there are stocks to sell from the last pass, sell them now
    for sell_stock in sell_portfolio:
        print(f"SELL: {sell_stock} on {processing_date}")
            
        # Create a TradeRecord object
        transact("Sell", processing_date, sell_stock, "Sell at the next open")          
            
    print("\n************************************************************")
    print(f"FINAL Portfolio : {today_portfolio}")   
    print("************************************************************")
                    
def query_distinct_dates(conn, table_name):
    """
    Queries a database table and returns distinct values in the specified date column, 
    ordered by date in ascending order.

    Args:
        conn (sqlite3.Connection): The connection object to the database.
        table_name (str): The name of the table to query.
        date_column (str): The name of the column containing date values.

    Returns:
        list: A list of distinct date strings ordered by date.
    """
    cursor = conn.cursor()

    # Use strftime('%Y-%m-%d') to format dates consistently for comparison
    # Adapt the format string if your dates are stored differently
    sql_statement = f"""
        SELECT DISTINCT "Last Run" 
        FROM {table_name} 
        ORDER BY 'Last Run' ASC;
    """

    cursor.execute(sql_statement)
    results = cursor.fetchall()
  
    # Extract only the date strings from the results
    distinct_dates = [row[0] for row in results]
    return distinct_dates

def get_eval_by_date(conn, datestr):
    """
    Queries the database for rows in the 'stock_data' table where the 'Last Run' 
    column matches the provided date string and returns a pandas DataFrame.

    Args:
        conn (sqlite3.Connection): The connection object to the database.
        datestr (str): The date string to filter by (format should match 'Last Run' column).

    Returns:
        pandas.DataFrame: A DataFrame containing the matching rows from the table, 
                            or an empty DataFrame if no data is found.
    """
    cursor = conn.cursor()

    sql_statement = f"""
        SELECT * FROM stock_data 
        WHERE "Last Run" = ?;
    """

    cursor.execute(sql_statement, (datestr,))  # Use tuple for parameter substitution
    results = cursor.fetchall()

    # Check if any results were found
    if results:
        # Convert results to a DataFrame using column names from cursor description
        df = pd.DataFrame(results, columns=[desc[0] for desc in cursor.description])
    else:
        # Return an empty DataFrame if no data is found
        df = pd.DataFrame()

    return df

def screen_for_buys(eval_df, ignore_supertrend_winners=False):
        if not ignore_supertrend_winners:
                buys_df = eval_df[ (eval_df['Supertrend Winner']==True) &  
                        (eval_df['Supertrend Result']=='Buy') & 
                        (eval_df['LR Next_Day Recomm'] == 'Buy,Buy,Buy') &
                        (eval_df['SMA Crossed_Up']=='Buy')].sort_values(by=['Supertrend Winner','Supertrend Result',
                                                                            'ST Signal_Date','SMA Crossed_Up','SMA_X_Date'],
                                                                        ascending=[False,True,False,True,False])
        else:
                buys_df = eval_df[ (eval_df['Supertrend Result']=='Buy') & 
                        (eval_df['LR Next_Day Recomm'] == 'Buy,Buy,Buy') &
                        (eval_df['SMA Crossed_Up']=='Buy')].sort_values(by=['Supertrend Winner','Supertrend Result',
                                                                            'ST Signal_Date','SMA Crossed_Up','SMA_X_Date'],
                                                                        ascending=[False,True,False,True,False])            
        
        return buys_df
    
def main():
    """Main function to connect, query, and process data"""
    db_file = "data.db"  # Replace with your actual database filename
    BuyBuyBuy = """
        select "Last Run", "Stock" , "Last Price" 
        from stock_data 
        where "LR Next_Day Recomm" = "Buy,Buy,Buy" 
        and "Supertrend Winner"=1 
        and "Supertrend Result"="Buy" 
        and "SMA Crossed_Up" = "Buy" 
        ORDER BY "Last Run";
    """

    SMA_X_Supertrend_Winner = """
        select "Last Run", "Stock" , "Last Price" 
        from stock_data 
        where "Supertrend Winner"=1 
        and "Supertrend Result"="Buy" 
        and "SMA Crossed_Up" = "Buy" 
        ORDER BY "Last Run";
    """
    All_Roads_Lead_UP_Safe = """
        select "Last Run", "Stock" , "Last Price" 
        from stock_data 
        where "Supertrend Winner"=1 
        and "Supertrend Result"="Buy" 
        and "SMA Crossed_Up" = "Buy" 
        ORDER BY "Last Run";
    """

    # Set sql_statement to one of the SQL queries defined above
    sql_statement = BuyBuyBuy 
    
    
    data = {}
    
    conn = connect_db(db_file)
    distinct_dates = query_distinct_dates(conn,'stock_data')
    
    datestr = '2024-05-10'
    df_evaluation = get_eval_by_date(conn, datestr)
    
    # Buy Buy and More Buy
    buys_eval_df= screen_for_buys(eval_df=df_evaluation,ignore_supertrend_winners=False)
    buys_eval_df = buys_eval_df.sort_values('Daily VaR',ascending=False).sort_values('%Sharpe Ratio',ascending=True).sort_values(by='SMA_X_Date', ascending=False)
    
    print("\nBUYS, BUYS, and more BUYS")
    print("=========================")
    print(f"{len(buys_eval_df)} Stocks:",flush=True)
    sublist = ','.join(buys_eval_df['Stock'].astype(str))
    print(sublist,flush=True)
    print(buys_eval_df)

    # Buy Buy and More Buy (Not necessarily ST winners)
    buys_eval_df2= screen_for_buys(eval_df=df_evaluation,ignore_supertrend_winners=True)
    buys_eval_df2 = buys_eval_df.sort_values('Daily VaR',ascending=False).sort_values('%Sharpe Ratio',ascending=True).sort_values(by='SMA_X_Date', ascending=False)
    
    print("\nBUYS, BUYS, and more BUYS (Not necessarily ST winners)")
    print("=========================")
    print(f"{len(buys_eval_df2)} Stocks:",flush=True)
    sublist_ = ','.join(buys_eval_df2['Stock'].astype(str))
    print(sublist_,flush=True)
    print(buys_eval_df2)


    # All Roads Lead to UP & Safe
    buys_safe = buys_eval_df[(buys_eval_df['Beta']>0.8) & (buys_eval_df['Beta']<2) ].sort_values(by='SMA_X_Date', ascending=False).sort_values('Daily VaR',ascending=False)

    print("\nAll Roads Lead to UP & Safe")
    print("======================++===")    
    print(f"{len(buys_safe)} Stocks:",flush=True)
    print(','.join(buys_safe['Stock'].astype(str)),flush=True)

    # print(sublist,flush=True)
    print(buys_safe)

    # SMA Crossed and in Supertrend & Winner
    Crossed_up = df_evaluation[(df_evaluation['SMA Crossed_Up'] == 'Buy') & (df_evaluation['Supertrend Result'] == 'Buy') & (df_evaluation['Supertrend Winner'] == True) ].sort_values('SMA_X_Date', ascending=False)
    sublist2 = ','.join(Crossed_up['Stock'].astype(str))
    
    print("\nSMA Crossed and in Supertrend & Winner")
    print("=====================+++++++++++++====")
    print(sublist2,flush=True)
    print(Crossed_up)
    
    # print(df_evaluation)
    print(distinct_dates)
    return
    
    cursor = query_data(conn, sql_statement)
    process_data(cursor)
    conn.close()
    
    # Print the data structure (accessing record fields using dot notation)
    old_date = None
    for record in transact_record:
        # print(f"Transaction: {record.transact}, Date: {record.date}, Stock: {record.stock}, Note: {record.note}")
        if record.date != old_date:
            print(f"On {record.date} :")
        print(f"\t{record.transact} {record.stock} ")
        add_data(data,record.date,f"{record.stock}")
        old_date = record.date

    records = []
    for record in transact_record:
        record_dict = {
            'Transact': record.transact,
            'Date': record.date,  # Assuming date is already a datetime object
            'Stock': record.stock,
            'Note': record.note
        }
        records.append(record_dict)

    # Assuming 'data' is the list of dictionaries from previous examples
    transact_df = pd.DataFrame(records)
    print(tabulate(transact_df, headers=transact_df.columns, tablefmt="grid"))
    
    # Convert dictionary to list of records (tuples)
    items = [(date, list(symbols)) for date, symbols in data.items()]
    buy_stocks_df = pd.DataFrame(items,columns=['Date','Stock'])
    buy_stocks_df = buy_stocks_df.set_index('Date')
    # print(buy_stocks_df)
    # Print the DataFrame
    print(tabulate(buy_stocks_df, headers=buy_stocks_df.columns, tablefmt="grid"))


    in_stock = input("Enter stock symbol : ").strip().upper()
    instock_list = [in_stock]
    # print("Target = ", instock_list)

    # Create a Series with 0s (assuming all dates initially don't have the stock)
    buy_alerts = pd.Series(0, index=buy_stocks_df.index)

    start_date = buy_alerts.index[0]
    # print(f"Alert start Date {start_date}")
    df_prices = yf.download(in_stock,start=start_date,interval='1d',progress=False)
    
    # print(df_prices)
    for i, row in buy_stocks_df.iterrows():
        # print(i,row["Stock"])
        if in_stock in row['Stock']:
            buy_alerts.loc[i] = 1
            # next_date_index = buy_alerts.index.get_loc(i) + 1  # Assuming index is a date object
            
            
            # if next_date_index < len(buy_alerts):  # Check if next index exists
            #     buy_alerts.loc[buy_alerts.index[next_date_index]] = 1
                # print(f"buy {in_stock} the next day {buy_alerts.index[next_date_index]} open")
                
        # if buy_alerts[i]:
            # print(f"{i} BUY {in_stock} at ${round(df_prices['Open'].loc[i],2)}")
            # print(f"{i} BUY {in_stock} ")
        
    buy_alerts_original = buy_alerts.copy()
    buy_alerts = buy_alerts.shift(1).fillna(0)
    # print("Buy Alerts ",buy_alerts)
    df_pred = pd.DataFrame(buy_alerts,columns=['Predicted'],index=buy_alerts_original.index)

    # Run backtesting on the model to verify the results
    backtest = bt(df_in=df_prices, signals=df_pred, start_date=start_date, end_date=None, amount=100000)
    backtest.run()
    tran_history = backtest.get_tran_history()
    # print(tran_history)
    backtest.results()
    backtest.plot_account(f"{in_stock} Backtest since {start_date}")



    # buy_alerts = buy_alerts.shift(1).fillna(0)
    # print(buy_alerts)
    
    # df_pred = pd.DataFrame(buy_alerts,columns=['Predicted'],index=df_lagged.index)

    # # Run backtesting on the model to verify the results
    # backtest = bt(df_in=df_in, signals=df_pred, start_date=start_date, end_date=None, amount=100000)
    # backtest.run()
    # tran_history = backtest.get_tran_history()
    
    
if __name__ == "__main__":
  main()
