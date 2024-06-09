import sqlite3
from datetime import date, datetime
import pandas as pd
import yfinance as yf
import ta_verifer as ta

    
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
    print(sql_statement)
    cursor.execute(sql_statement)

    return cursor

def update_record(conn,date_str,stock,new_ema_trend, new_fastslow, new_lookback):
    
    cursor = conn.cursor()
    update_stmt = f"""
    UPDATE "stock_data" 
    SET "EMA_Trend" = '{new_ema_trend}', "EMA_FasSlow" = '{new_fastslow}', "EMA_Lookback" = '{float(new_lookback)}'
    WHERE "Stock" = '{stock}' AND "Last Run" = '{date_str}' 
    """
    print(update_stmt)
    cursor.execute(update_stmt)
    conn.commit()   
    cursor.close()
    
    return

def get_rows_by_stock(conn,stock):
    select_stmt = f"select * from 'stock_data' where Stock = '{stock}' order by 'Last_Run' asc"
    cursor = query_data(conn=conn,sql_statement=select_stmt)
    
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=[desc[0] for desc in cursor.description])
    
    cursor.close()
    
    return df

def fix_EMA_in_db(in_stock,df,fast=20,slow=40,lookback=7):
    if df == None:
        df = yf.download(in_stock,start="2023-01-01",period='1d',progress=False)
    
    last_date, current_recommendation,df_ema_recommendations = ta.calculate_pdta_alphatrend(stock=in_stock,df=df,Fast_EMA=fast,Slow_EMA=slow,Lookback=lookback)
    
    conn = connect_db(db_file='data.db')
    
    for idx, row in df_ema_recommendations.iterrows():
        if row['Buy_Sell_Signal'] == 1:
            new_ema_trend = 'Buy'
        elif row['Buy_Sell_Signal'] ==0:
            new_ema_trend = 'Sell'
            
        update_record(conn=conn,date_str=idx.strftime('%Y-%m-%d'),stock=in_stock,new_ema_trend=new_ema_trend,new_fastslow=f"{fast}X{slow}",new_lookback=lookback )
    
if __name__=="__main__":
    conn = connect_db(db_file='data.db')
    
    # update_record(conn=conn,date_str='2024-03-25',stock='AAPL',new_ema_trend='Banana',new_fastslow='20X40',new_lookback=7 )
    stock = input("Enter Stock Symbol :").strip().upper()
    start_date =   input("  Start Date (YYYY-MM-DD);")
    Fast_EMA = int(input("  Fast EMA (Short interval):"))
    Slow_EMA = int(input("  Slow EMA (Long interval):"))
    Lookback = int(input("  Lookback Period :"))
    
    df = None
    
    fix_EMA_in_db(stock,df,fast=Fast_EMA,slow=Slow_EMA,lookback=Lookback)
    df_updated = get_rows_by_stock(conn=conn,stock=stock)
    
    print(df_updated)
