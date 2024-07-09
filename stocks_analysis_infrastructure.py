import sys
import os
import pymysql as mysql
from IPython.display import display, HTML, Markdown, Image

# Add the directory to sys.path
script_directory = os.path.expanduser('~/Development/stocks_evaluator')
if script_directory not in sys.path:
    sys.path.append(script_directory)

import ta_verifer as ta
import yfinance as yf
import pandas as pd
import ema_indicator_lib as ema
import buy_sell_signal_generator as signals

def connect_db(host='localhost', user='root', password='Down2earth!', database='mystocks'):
    """Connects to the MySQL database and returns the connection object.

    Args:
        host (str): The hostname or IP address of the MySQL server.
        user (str): The username to use when connecting to the database.
        password (str): The password to use when connecting to the database.
        database (str): The name of the database to connect to.

    Returns:
        pymysql.Connection: The connection object to the database.
    """
    import pymysql 
    conn = pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    return conn

def get_stock_data_by_date(date_str):
    
    sql_statement = """
        SELECT * FROM stock_data 
        WHERE Last_Run = %s;
    """
    conn = connect_db()
    cursor = conn.cursor()
    
    cursor.execute(sql_statement, (date_str,))  # Use tuple for parameter substitution
    results = cursor.fetchall()
    # Check if any results were found
    if results:
        # Convert results to a DataFrame using column names from cursor description
        df = pd.DataFrame(results, columns=[desc[0] for desc in cursor.description])
    else:
        # Return an empty DataFrame if no data is found
        df = pd.DataFrame()

    
    return df

if __name__ == "__main__":
    pass