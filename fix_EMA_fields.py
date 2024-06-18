import sqlite3
from datetime import date, datetime
import pandas as pd
import yfinance as yf
import ta_verifer as ta

import pymysql
import time

def connect_db_mysql(host='localhost', user='root', password='Down2earth!', database='mystocks'):
    """Connects to the MySQL database and returns the connection object.

    Args:
        host (str): The hostname or IP address of the MySQL server.
        user (str): The username to use when connecting to the database.
        password (str): The password to use when connecting to the database.
        database (str): The name of the database to connect to.

    Returns:
        pymysql.Connection: The connection object to the database.
    """
    conn = pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    return conn
 
def query_data(conn, sql_statement):
    """Executes the provided SQL statement and returns the cursor object.

    Args:
        conn (pymysql.Connection): The connection object to the database.
        sql_statement (str): The SQL statement to be executed.

    Returns:
        pymysql.cursors.Cursor: The cursor object containing the results of the query.
    """
    cursor = conn.cursor()
    cursor.execute(sql_statement)
    return cursor

def get_stocks_list_from_db(conn):
    sql_statement = "SELECT DISTINCT Stock from stock_data ORDER BY Stock ASC"
    cursor = query_data(conn=conn,sql_statement=sql_statement)
    stocks_list = cursor.fetchall()
    print(stocks_list)
    return stocks_list
    
    
def connect_db_sqlite3(db_file):
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

def update_momentum_record(conn,date_str,stock,mom_str,confirm_str,days_in_confirm):
    
    cursor = conn.cursor()
    update_stmt = f"""
    UPDATE stock_data
    SET Momentum = '{mom_str}', Confirmation = '{confirm_str}', Mom_Days_Confirmed = {days_in_confirm}
    WHERE Stock = '{stock}' AND Last_Run = '{date_str}' 
    """
    print(update_stmt)
    cursor.execute(update_stmt)
    conn.commit()   
    cursor.close()
    
    return


def update_record(conn,date_str,stock,new_ema_trend, new_fastslow, new_lookback,ema_days_at_buy):
    
    cursor = conn.cursor()
    update_stmt = f"""
    UPDATE stock_data
    SET EMA_Trend = '{new_ema_trend}', EMA_FastXSlow = '{new_fastslow}', EMA_Lookback = {new_lookback}, EMA_Days_at_Buy = {ema_days_at_buy}
    WHERE Stock = '{stock}' AND Last_Run = '{date_str}' 
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
    
    last_date, current_recommendation,df_ema_recommendations, ema_days_at_buy  = ta.calculate_pdta_alphatrend(stock=in_stock,df=df,Fast_EMA=fast,Slow_EMA=slow,Lookback=lookback)
    
    conn_sqlite3 = connect_db_sqlite3(db_file='data.db')
    conn_mysql = connect_db_mysql()
    
    for idx, row in df_ema_recommendations.iterrows():
        if row['Buy_Sell_Signal'] == 1:
            new_ema_trend = 'Buy'
        elif row['Buy_Sell_Signal'] ==0:
            new_ema_trend = 'Sell'
            
        update_record(conn=conn_sqlite3,date_str=idx.strftime('%Y-%m-%d'),stock=in_stock,new_ema_trend=new_ema_trend,new_fastslow=f"({fast}X{slow})",new_lookback=lookback,ema_days_at_buy=ema_days_at_buy )
        update_record(conn=conn_mysql,date_str=idx.strftime('%Y-%m-%d'),stock=in_stock,new_ema_trend=new_ema_trend,new_fastslow=f"({fast}X{slow})",new_lookback=lookback,ema_days_at_buy=ema_days_at_buy )

def fix_Momentum_in_db(in_stock,df,fast=20,slow=40,lookback=7):
    if df == None:
        df = yf.download(in_stock,start="2023-01-01",period='1d',progress=False)
    
    last_date, current_recommendation,df_ema_recommendations, ema_days_at_buy  = ta.calculate_pdta_alphatrend(stock=in_stock,df=df,Fast_EMA=fast,Slow_EMA=slow,Lookback=lookback)

    df_trend_pdta = ta.get_momentum_indicators(df_ema_recommendations)
    # print(df_trend_pdta[['Buy_Sell_Signal','Mom_Signal']])
    # print(df_trend_pdta.columns)
    
    df_trend_pdta['Confirmation'] = 0
    
    for idx,row in df_trend_pdta.iterrows():
        if  df_trend_pdta.loc[idx,'Buy_Sell_Signal'] == 1 and df_trend_pdta.loc[idx,'Mom_Signal'] == 1 :
            df_trend_pdta.loc[idx,'Confirmation'] = 1
        else:
            df_trend_pdta.loc[idx,'Confirmation'] = -1
    
    conn_sqlite3 = connect_db_sqlite3(db_file='data.db')
    conn_mysql = connect_db_mysql()
    
    # Determine if Momentum is confirmed if both signals are 1
    # last_momentum = df_trend_pdta['Mom_Signal'].index[-1]
    # last_ema_trend = df_trend_pdta['Buy_Sell_Signal'].index[-1]
        
    for idx,row in  df_trend_pdta.iterrows():
        consecutive_confirm_days = 0
        
        if df_trend_pdta.loc[idx,'Mom_Signal'] == 1 :
            last_momentum_str = 'Buy'
        else: 
            last_momentum_str = 'Sell'
        if df_trend_pdta.loc[idx,'Confirmation'] == 1:
            confirmation_str = 'Buy'
            
            # Find the number of days in 'Buy' recommendations
            # print(idx)
            for date in reversed(df_trend_pdta.index[df_trend_pdta.index <= idx]):
                if df_trend_pdta.loc[date, 'Buy_Sell_Signal'] == 1 & df_trend_pdta.loc[date,'Mom_Signal'] == 1:
                    # date_str2 = date.strftime('%Y-%m-%d')
                    
                    consecutive_confirm_days += 1
                    # print(f"Buy Confirmed on {date_str2}, Days in Buy = {consecutive_confirm_days}")                
                else:
                    break              
        else:
            confirmation_str = 'Sell'
            
        print(f" UPDATING: :\n\t date_str: {idx.strftime('%Y-%m-%d')}\n\tStock:{in_stock}\n\tmomentum_str:{last_momentum_str} \n\tconfirm_str:{confirmation_str}\n\tDays in Confirmation:{consecutive_confirm_days}\n")
        
        

    # consecutive_confirm_days = 0
    # # if both signals are 1 then we have confirmation
    # if last_ema_trend == 1 and last_momentum == 1:
    #     last_confirmation_str = 'Buy'
    #     # Carculate days count in confirmed momentum 
            
    #     # Find the number of days in 'Buy' recommendations
    #     for date in reversed(df_trend_pdta.loc[last_date]):
    #         # date_str = date.strftime('%Y-%m-%d')
    #         if df_trend_pdta.loc[date, 'Buy_Sell_Signal'] == 1 & df_trend_pdta.loc[date,'Mom_Signal'] == 1:
    #             consecutive_confirm_days += 1
    #         else:
    #             break            
    # else:
    #     last_confirmation_str = 'Sell'
    #     consecutive_confirm_days = -1
                
    # print(f"***Stock={in_stock} = {last_date}\nlast_momentum_str={last_momentum_str}\nlast_confirmation_str={last_confirmation_str}\nconsecutive_confirm_days={consecutive_confirm_days}")


    # update_momentum_record(conn_mysql,date_str=idx.strftime('%Y-%m-%d'),stock=in_stock,confirm_str=row[''],days_in_confirm)

if __name__=="__main__":
    conn = connect_db_mysql()
    # conn = connect_db_sqlite3(db_file='data.db')
    
    # update_record(conn=conn,date_str='2024-03-25',stock='AAPL',new_ema_trend='Banana',new_fastslow='20X40',new_lookback=7 )
    # stock = input("Enter Stock Symbol :").strip().upper()
    # start_date =   input("  Start Date (YYYY-MM-DD);")
    # Fast_EMA = int(input("  Fast EMA (Short interval):"))
    # Slow_EMA = int(input("  Slow EMA (Long interval):"))
    # Lookback = int(input("  Lookback Period :"))
    
    df = None
    start_date =  "2024-03-23"
    Fast_EMA = 20
    Slow_EMA = 40
    Lookback = 7

    stocks_list = get_stocks_list_from_db(conn)
    fix_Momentum_in_db('AAPL',df)
    # for i in range(len(stocks_list)):
    #     stock = stocks_list[i][0]
        
    #     # fix_EMA_in_db(stock,df,fast=Fast_EMA,slow=Slow_EMA,lookback=Lookback)
        
    #     fix_Momentum_in_db(stock,df)
        
    #     time.sleep(0.5)
        
    # df_updated = get_rows_by_stock(conn=conn,stock=stock)
    # print(df_updated)
