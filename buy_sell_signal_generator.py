import sqlite3
from datetime import date, datetime

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
    
    processing_date = datetime.strptime("2000-01-01", "%Y-%m-%d")
    for row in cursor.fetchall():
        last_run, stock, last_price = row
        date_last_run =  datetime.strptime(last_run, "%Y-%m-%d")
        stock = stock.strip()
        if date_last_run > processing_date:       
            print("************************************************************")
            print(f"On({processing_date}) LATEST Portfolio : {current_portfolio}")   
            print("************************************************************")
            processing_date = date_last_run
            print(f"New Date: {processing_date}")
            
            for sell_stock in set(current_portfolio):
                if sell_stock != stock:
                    print(f"SELL: {sell_stock} on {last_run}")
                    del current_portfolio[sell_stock]
                
            if stock not in set(current_portfolio):
                print(f"BUY: {stock} on {last_run} at ${last_price}")
                current_portfolio[stock] = last_run
            
            today_portfolio = {}
            today_portfolio[stock] = last_run
            
        else:
            today_portfolio[stock] = last_run
            
            if stock not in set(current_portfolio):
                print(f"BUY: {stock} on {last_run} at ${last_price}")
                current_portfolio[stock] = last_run


    print("\n************************************************************")
    print(f"FINAL Portfolio : {current_portfolio}")   
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

if __name__ == "__main__":
  main()
