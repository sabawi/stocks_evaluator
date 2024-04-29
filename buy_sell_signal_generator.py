import sqlite3
from datetime import date, datetime
import pandas as pd
from tabulate import tabulate

transact_record = []

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


    print("\n************************************************************")
    print(f"FINAL Portfolio : {today_portfolio}")   
    print("************************************************************")
                    


def main():
    """Main function to connect, query, and process data"""
    db_file = "data.db"  # Replace with your actual database filename
    sql_statement = """
        select "Last Run", "Stock" , "Last Price" 
        from stock_data 
        where "LR Next_Day Recomm" = "Buy,Buy,Buy" 
        and "Supertrend Winner"=1 
        and "Supertrend Result"="Buy" 
        and "SMA Crossed_Up" = "Buy" 
        ORDER BY "Last Run";
    """

    conn = connect_db(db_file)
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
    df = pd.DataFrame(records)

    # Print the DataFrame
    # print(tabulate(df, headers=df.columns, tablefmt="grid"))

if __name__ == "__main__":
  main()
