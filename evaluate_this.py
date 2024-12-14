# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm
from scipy.stats import gstd as geo_std
import datetime
import os
import supertrend_lib as stlib
import sbeta_lib as sbeta
import smacross_lib as smacross
import CAPM_lib as capm
import LR_run_model_lib as LR_run
import LR_model_lib as LR_models
import ta_verifer as ema_trend_ind
import ema_indicator_lib as ema_lib
import yfinance as yf
pd.set_option('display.max_rows', 200)
from IPython.display import display, HTML
import sys

# Flush all print outputs in this script
sys.stdout.flush()

# %%
# symbols_list = eval_df.Stock.values
def get_company_info(symbols_list):
    info_list = []

    for s in symbols_list:
        shandle = yf.Ticker(s)
        info_list.append(shandle.info)

    return info_list
    
# info_list = get_company_info(['aapl','msft'])
# print(info_list)

# %%
def add_update(eval_df,values):
    """ Add a a row to dataframe from a list of data points in the same sequence as the columns

    Args:
        eval_df (_type_): DataFrame receiving the new row
        values_list (_type_): A list of values in the same data types and sequence order as the dataframe columns

    Returns:
        _type_: The updated dataframe
    """
    tmp_df = pd.DataFrame([values], columns=eval_df.columns)
    eval_df = pd.concat([eval_df, tmp_df], ignore_index=True)
    return eval_df

# %%
def read_stocklist(filename):
    with open(filename) as f:
        stocklist = f.read()
    return stocklist.strip().split(",")


# %%
def create_directory(dir_path):
    # Get the user's home directory
    home_directory = os.path.expanduser('~')
    
    # Define the path for the subdirectory
    subdirectory_path = os.path.join(home_directory, dir_path)
          
    # Check if the directory already exists
    if not os.path.exists(subdirectory_path):
        # If it doesn't exist, create the directory
        os.makedirs(subdirectory_path)
        
    return subdirectory_path

# %%
def save_list_to_file(sublist,dir_path='.',list_filename='unamed_list.csv'):
    subdirectory_path = create_directory(dir_path)
    full_filename = os.path.join(subdirectory_path,list_filename)
    print(f"Saving {full_filename} to disk ...",end='',flush=True)
    with open( full_filename, 'w') as f:
        f.write(sublist)
        print(f'"{full_filename}" updated!',flush=True)
    
    print("...Done!",flush=True)


# import mysql.connector as mysqlpy
import pymysql as mysqlpy

def create_database():
    # Connect to the MySQL database
    conn = mysqlpy.connect(
        host='localhost',
        user='root',
        password='Down2earth!',
        database='mystocks'
    )
    cursor = conn.cursor()

    # Define table schema with composite primary key
    cursor.execute('''CREATE TABLE IF NOT EXISTS stock_data (
                        `Last_Run` VARCHAR(255),
                        `Stock` VARCHAR(255),
                        `Last_Price` DOUBLE,
                        `Std._Dev.%` VARCHAR(255),
                        `$Std._Dev.` VARCHAR(255),
                        `Supertrend_Winner` TINYINT(1),
                        `Supertrend_Result` VARCHAR(255),
                        `ST_Signal_Date` VARCHAR(255),
                        `Days@ST` DOUBLE,
                        `LR_Best_Model` VARCHAR(255),
                        `LR_Next_Day_Recomm` VARCHAR(255),
                        `SMA_Crossed_Up` VARCHAR(255),
                        `SMA_X_Date` VARCHAR(255),
                        `SMA_FastXSlow` VARCHAR(255),
                        `Beta` DOUBLE,
                        `Sharpe_Ratio%` DOUBLE,
                        `CAPM` DOUBLE,
                        `Daily_VaR` DOUBLE,
                        `EMA_Trend` VARCHAR(255),
                        `EMA_Days_at_Buy` INT,
                        `EMA_FastXSlow` VARCHAR(255),
                        `EMA_Lookback` INT,
                        PRIMARY KEY(`Last_Run`, `Stock`)
                    );
                ''')

    # Commit changes and close connection
    conn.commit()
    conn.close()


# %% [markdown]
# ### Create an SQLite Database

# %%
import sqlite3

# Function to create an SQLite database and table with a composite primary key
def create_database_old():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    # Define table schema with composite primary key
    cursor.execute('''CREATE TABLE "stock_data" (
                    "Last_Run"	TEXT,
                    "Stock"	TEXT,
                    "Last_Price"	REAL,
                    "Std._Dev.%"	TEXT,
                    "$Std._Dev."	TEXT,
                    "Supertrend_Winner"	BOOLEAN,
                    "Supertrend_Result"	TEXT,
                    "ST_Signal_Date"	TEXT,
                    "Days@ST"	REAL,
                    "LR_Best_Model"	TEXT,
                    "LR_Next_Day_Recomm"	TEXT,
                    "SMA_Crossed_Up"	TEXT,
                    "SMA_X_Date"	TEXT,
                    "SMA_FastXSlow"	TEXT,
                    "Beta"	REAL,
                    "Sharpe_Ratio%"	REAL,
                    "CAPM"	REAL,
                    "Daily_VaR"	REAL,
                    "EMA_Trend"	TEXT,
                    "EMA_Days_at_Buy" INTEGER,
                    "EMA_FastXSlow"	TEXT,
                    "EMA_Lookback"	INTEGER,
                    PRIMARY KEY("Last_Run","Stock")
                );
                    )''')
    conn.commit()
    conn.close()


# create_database()

# %% [markdown]
# ### Insert Dataframe Data into Database 



def insert_data_into_database_error(df):
    # Make a copy of the DataFrame to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Convert date/time columns to the appropriate format
    df_copy['Last_Run'] = pd.to_datetime(df_copy['Last_Run']).dt.strftime('%Y-%m-%d')
    df_copy['ST_Signal_Date'] = pd.to_datetime(df_copy['ST_Signal_Date']).dt.strftime('%Y-%m-%d')
    df_copy['SMA_X_Date'] = pd.to_datetime(df_copy['SMA_X_Date']).dt.strftime('%Y-%m-%d')

    # Convert boolean columns to integers
    df_copy['Supertrend_Winner'] = df_copy['Supertrend_Winner'].astype(int)
    df_copy['Days@ST'] = df_copy['Days@ST'].astype(int)
    df_copy['EMA_Days_at_Buy'] = df_copy['EMA_Days_at_Buy'].astype(int)
    df_copy['EMA_Lookback'] = df_copy['EMA_Lookback'].astype(int)

    # Print the DataFrame column names for debugging
    # print("DataFrame columns:", df_copy.columns)

    # Connect to the MySQL database
    conn = mysqlpy.connect(
        host='localhost',
        user='root',
        password='Down2earth!',
        database='mystocks'
    )

    cursor = conn.cursor()

    # Define column names
    columns = list(df_copy.columns)

    formatted_columns = ', '.join([f'"{col}"' for col in columns])

    num_placeholders = len(columns)
    placeholders = ', '.join(['%s'] * num_placeholders)

    query = f'''INSERT INTO stock_data ({', '.join(columns)})
                VALUES ({placeholders})'''

    # print("Generated Query:", query)  # Debugging statement

    # Iterate through each row in the DataFrame
    for index, row in df_copy.iterrows():
        # print(f"\nRow from the DataFrame: {row}\n")
        # Check if the combination of 'Last_Run' and 'Stock' already exists in the table
        cursor.execute('''SELECT COUNT(*) FROM stock_data WHERE `Last_Run` = %s AND `Stock` = %s''',
                       (row['Last_Run'], row['Stock']))
        result = cursor.fetchone()[0]

        # If the combination already exists, skip insertion
        if result > 0:
            print(f"Skipping insertion for row {index+1}: Last_Run {row['Last_Run']} and Stock {row['Stock']} already exists.", flush=True)
        else:
            def escape_special_chars(data):
                """Escapes special characters in data for safe insertion into the query"""
                if isinstance(data, str):
                    return data.replace("'", "\\'")  # Escape single quotes
                else:
                    return data

            data_list = [escape_special_chars(row) for row in df_copy.to_records(index=False)]  # Escape and create data list

            # ... rest of your code with data_list

            query = f'''INSERT INTO stock_data ({', '.join(columns)})
                        VALUES (%s, %s, ..., %s)'''  # Use ... for placeholders

            # Prepare data tuple
            # data = tuple(row[col] for col in columns)

            # print(f"Final Query:\n{query}\nData:{data_list}")
            # Insert data using cursor.execute
            # cursor.execute(query, data)
            cursor.executemany(query, data_list)
        

    # Commit changes and close connection
    conn.commit()
    conn.close()



def insert_data_into_database(df):
    
    print("Inserting data into MySQL Database table....",end='',flush=True)
    # Make a copy of the DataFrame to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Convert date/time columns to the appropriate format
    df_copy['Last_Run'] = pd.to_datetime(df_copy['Last_Run']).dt.strftime('%Y-%m-%d')
    df_copy['ST_Signal_Date'] = pd.to_datetime(df_copy['ST_Signal_Date']).dt.strftime('%Y-%m-%d')
    df_copy['SMA_X_Date'] = pd.to_datetime(df_copy['SMA_X_Date']).dt.strftime('%Y-%m-%d')

    # Convert boolean columns to integers
    df_copy['Supertrend_Winner'] = df_copy['Supertrend_Winner'].astype(int)
    df_copy['Days@ST'] = df_copy['Days@ST'].astype(int)
    df_copy['EMA_Days_at_Buy'] = df_copy['EMA_Days_at_Buy'].astype(int)
    df_copy['EMA_Lookback'] = df_copy['EMA_Lookback'].astype(int)

    # Convert boolean columns to integers
    df_copy['Supertrend_Winner'] = df_copy['Supertrend_Winner'].astype(int)

    # Connect to the MySQL database
    conn = mysqlpy.connect(
        host='localhost',
        user='root',
        password='Down2earth!',
        database='mystocks'
    )

    cursor = conn.cursor()

    # Iterate through each row in the DataFrame
    for index, row in df_copy.iterrows():
        # Check if the combination of 'Last_Run' and 'Stock' already exists in the table
        cursor.execute('''SELECT COUNT(*) FROM stock_data WHERE `Last_Run` = %s AND `Stock` = %s''',
                       (row['Last_Run'], row['Stock']))
        result = cursor.fetchone()[0]

        # If the combination already exists, skip insertion
        if result > 0:
            print(f"Skipping insertion for row {index+1}: Last_Run {row['Last_Run']} and Stock {row['Stock']} already exists.", flush=True)
        else:
            # Define column names and data as separate lists
            column_names = df_copy.columns.tolist()
            data = tuple(row)

            # Print the length of column_names and a sample data list to compare
            # print(f"Number of columns: {len(column_names)}")
            # print(f"Sample data list: {data[:5]}")  # Print only the first 5 elements

            column_list = f"""`Last_Run`, `Stock`, `Last_Price`, `Std._Dev.%`, `$Std._Dev.`,
                            `Supertrend_Winner`, `Supertrend_Result`, `ST_Signal_Date`, `Days@ST`,
                            `LR_Best_Model`, `LR_Next_Day_Recomm`, `SMA_Crossed_Up`, `SMA_X_Date`,
                            `SMA_FastXSlow`, `Beta`, `Sharpe_Ratio%`, `CAPM`, `Daily_VaR`,
                            `EMA_Trend`, `EMA_Days_at_Buy`, `EMA_FastXSlow`, `EMA_Lookback`,
                            `Momentum`, `Confirmation`, `Mom_Days_Confirmed`"""
            data_str =    f""" "{row['Last_Run']}", "{row['Stock']}", {row['Last_Price']}, "{row['Std._Dev.%']}", "{row['$Std._Dev.']}",
                            {row['Supertrend_Winner']}, "{row['Supertrend_Result']}", "{row['ST_Signal_Date']}", {row['Days@ST']},
                            "{row['LR_Best_Model']}", "{row['LR_Next_Day_Recomm']}", "{row['SMA_Crossed_Up']}", "{row['SMA_X_Date']}",
                            "{row['SMA_FastXSlow']}", {row['Beta']}, {row['Sharpe_Ratio%']}, {row['CAPM']}, {row['Daily_VaR']},
                            "{row['EMA_Trend']}", {row['EMA_Days_at_Buy']},"{row['EMA_FastXSlow']}", {int(row['EMA_Lookback'])},
                            "{row['Momentum']}","{row['Confirmation']}", {int(row['Mom_Days_Confirmed'])}"""
            # Prepare parameterized INSERT query
            query = f"INSERT INTO stock_data ("+column_list+") VALUES (" +data_str + ");"
            # print(query)
            # Insert data using cursor.execute
            cursor.execute(query)

    # Commit changes and close connection
    conn.commit()
    conn.close()     
    print("... Done!",flush=True)
      
import sqlite3

def insert_data_into_database_old(df):
    print("Inserting data into SQLite3 Database table....",end='',flush=True)
    
    # Make a copy of the DataFrame to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Convert date/time columns to the appropriate format
    df_copy['Last_Run'] = pd.to_datetime(df_copy['Last_Run']).dt.strftime('%Y-%m-%d')
    df_copy['ST_Signal_Date'] = pd.to_datetime(df_copy['ST_Signal_Date']).dt.strftime('%Y-%m-%d')
    df_copy['SMA_X_Date'] = pd.to_datetime(df_copy['SMA_X_Date']).dt.strftime('%Y-%m-%d')

    # Convert boolean columns to integers
    df_copy['Supertrend_Winner'] = df_copy['Supertrend_Winner'].astype(int)

    # Connect to the SQLite database
    conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), 'data.db'))

    cursor = conn.cursor()

    # Iterate through each row in the DataFrame
    for index, row in df_copy.iterrows():
        # Check if the combination of 'Date' and 'Stock' already exists in the table
        cursor.execute('''SELECT COUNT(*) FROM stock_data WHERE "Last_Run" = ? AND "Stock" = ?''',
                       (row['Last_Run'], row['Stock']))
        result = cursor.fetchone()[0]

        # If the combination already exists, skip insertion
        if result > 0:
            print(f"SQLite3 Skipping insertion for row {index+1}: Date {row['Last_Run']} and Stock {row['Stock']} already exists.", flush=True)
        else:
            # Print out the row values before insertion
            # print(f"Inserting row {index+1}: {row}")

            # Convert problematic columns to string if necessary
            row['SMA_X_Date'] = str(row['SMA_X_Date'])
            row['SMA_FastXSlow'] = str(row['SMA_FastXSlow'])
            row['EMA_Trend'] = str(row['EMA_Trend'])
            row['EMA_FastXSlow'] = str(row['EMA_FastXSlow'])
            
            # Insert the row into the table
            cursor.execute('''INSERT INTO stock_data (
                                `Last_Run`, `Stock`, `Last_Price`, `Std._Dev.%`, `$Std._Dev.`, 
                                `Supertrend_Winner`, `Supertrend_Result`, `ST_Signal_Date`, 
                                `Days@ST`, `LR_Best_Model`, `LR_Next_Day_Recomm`, `SMA_Crossed_Up`, 
                                `SMA_X_Date`, `SMA_FastXSlow`, `Beta`, `Sharpe_Ratio%`, `CAPM`, 
                                `Daily_VaR`, `EMA_Trend`, `EMA_FastXSlow`, `EMA_Lookback`, `EMA_Days_at_Buy`, 
                                `Momentum`, `Confirmation`, `Mom_Days_Confirmed`
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                           (row['Last_Run'], row['Stock'], row['Last_Price'], row['Std._Dev.%'], row['$Std._Dev.'],
                            row['Supertrend_Winner'], row['Supertrend_Result'], row['ST_Signal_Date'], row['Days@ST'],
                            row['LR_Best_Model'], row['LR_Next_Day_Recomm'], row['SMA_Crossed_Up'], row['SMA_X_Date'],
                            row['SMA_FastXSlow'], row['Beta'], row['Sharpe_Ratio%'], row['CAPM'], row['Daily_VaR'],
                            row['EMA_Trend'], row['EMA_FastXSlow'], row['EMA_Lookback'], row['EMA_Days_at_Buy'],
                            row['Momentum'], row['Confirmation'], row['Mom_Days_Confirmed']))

    # Commit changes and close connection
    conn.commit()
    conn.close()
    print("Done!",flush=True)
    
# Example usage:
# insert_data_into_database(eval_df)

# %% [markdown]
# ### Write Dataframe as Table to HTML File

def make_text_clickable(in_df, column_name, linktext, replace_text):
    out_df = in_df.copy()
    # pd.options.mode.copy_on_write = True
    
    for index,row in in_df.iterrows():
        old_text = out_df.loc[index,column_name]
        new_text = linktext.replace(replace_text, old_text)
        new_text = f"<a href='{new_text}'  target='_blank' title='{old_text} External Link'>{old_text}</a>"
        out_df.loc[index,column_name] = new_text
        
    return out_df

# %%
from prettytable import PrettyTable

def df_to_html_file(input_df, output_file_name, table_title):
    
    print(f"Saving HTML file {output_file_name} ...",end='',flush=True)
    # Create a PrettyTable object
    table = PrettyTable(input_df.columns.tolist())

    # Add rows to the table
    for _, row in input_df.iterrows():
        table.add_row(row.tolist())

    # Generate the HTML table
    table_text = table.get_html_string()

    # Make a comma separated list of stocks in one line
    # linktext = "https://finviz.com/quote.ashx?t=????&p=d"
    linktext = "https://digital.fidelity.com/prgw/digital/research/quote/dashboard/summary?symbol=????"
    input_df1 = input_df.copy()
    input_df1 = make_text_clickable(input_df1,'Stock',linktext,'????')
    
    stock_list = ', '.join(input_df1['Stock'].astype(str))
    
    # Manually set the table ID to "sortable-table"
    table_text = table_text.replace("<table>", '<table id="sortable-table">')

    page_header = ''' 
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
    '''

    page_header = f"{page_header}<title>{table_title}</title>"
    # Add table title
    page_header = f"{page_header}\n   <h2>{table_title}</h2>\n"

    # Add CSS styling
    css_style = '''
        <style>
        table {
            border-collapse: collapse;
            width: 100%;
        }
        th, td {
            border: 1px solid black;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        </style>
    '''

    # Include DataTables JavaScript and CSS
    scripts = '''
    
        <script src="https://code.jquery.com/jquery-3.6.4.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
        
        <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">

        <script>
            $(document).ready( function () {
                $('#sortable-table').DataTable({"lengthMenu": [ [10, 25, 50, -1], [10, 25, 50, "All"] ], // Customize the options as needed
                    "pageLength": -1 // Change this number to set the initial number of rows displayed per page
                    });
            });
        </script>
    '''

    # Combine CSS styling and table content
    html_page = page_header + css_style +"\n" + scripts + "\n" + "</head>" +"\n"

    # Wrap the table with a div for DataTables
    html_page = f'{html_page}\n<div class="dataTables_wrapper">\n{table_text}\n</div>\n<div>{stock_list}</div>\n</html>'

    # Save the HTML table to the file
    with open(output_file_name, 'w') as f:
        f.write(html_page)
        
    print("...Done!",flush=True)


# %% [markdown]
# ### Write/Append Dataframe to CSV File

# %%
def append_to_csv(df, file_path):
    # Append DataFrame to CSV file
    df.to_csv(file_path, mode='a', header=False, index=False)


def write_to_csv(df, csv_file="Historical_Eval_Runs.csv"):
  """
  Appends a DataFrame to a CSV file, handling headers, duplicates, and sorting.

  Args:
      df (pandas.DataFrame): The DataFrame to append.
      csv_file (str): The path to the CSV file.
  """
  exists = os.path.exists(csv_file)

  # Read existing data (if any)
  if exists:
    existing_df = pd.read_csv(csv_file, skiprows=1, parse_dates=True)
    # Set index only if existing_df is not empty
    if not existing_df.empty:
      existing_df.set_index('Last_Run', inplace=True)  # Set index after reading
  else:
    existing_df = pd.DataFrame()

  # Extract column names and data types from the DataFrame (unchanged)
  columns = df.columns.tolist()
  dtypes = df.dtypes.to_dict()

  # Remove duplicates based on 'Last_Run'
  df.drop_duplicates(subset='Last_Run', inplace=True)
  existing_df.drop_duplicates(subset='Last_Run', inplace=True)

  # Combine DataFrames
  combined_df = pd.concat([existing_df, df], sort=False)

  # Sort by 'Last_Run' (ascending)
  combined_df = combined_df.sort_values(by='Last_Run')

  # Write to CSV file
  combined_df.to_csv(csv_file, mode='a' if exists else 'w', index=False, header=not exists)


# %%
def update_models(stock_list):
    ret = LR_models.update_list_of_models(stock_list=stock_list)
    models_df = pd.DataFrame({'Stock' : stock_list,
                              'Model' : [f"Model{i}" for i in ret]})
    
    # print(models_df)
    return models_df

# %%
# stocks_from_file = read_stocklist(filename='./stocks_list.txt')
# stocks_from_file = [s.strip().upper() for s in list(stocks_from_file)]
# dfff = update_models(stocks_from_file)
# dfff

# %%
# dfff[dfff['Stock'] == 'AAPL']['Model'].values[0]

# %%
def run_supertrend(stock,start_date):
    return stlib.supertrend(stock,start_date=start_date)


# %%
def init_eval_table():
    columns = ['Last_Run'          ,'Stock'             ,'Last_Price'     ,'Std._Dev.%','$Std._Dev.','Supertrend_Winner' ,'Supertrend_Result','ST_Signal_Date', 'Days@ST',
               'LR_Best_Model'     ,'LR_Next_Day_Recomm', 'SMA_Crossed_Up','SMA_X_Date'      ,'SMA_FastXSlow' , 'EMA_Trend', 'EMA_Days_at_Buy','EMA_FastXSlow', 'EMA_Lookback',
               'Momentum'          , 'Confirmation'     ,'Mom_Days_Confirmed', 'Beta'          , 'Sharpe_Ratio%'                  ,'CAPM'              ,'Daily_VaR'  ]

    dtypes = ['datetime64[ns]'     , str                , float            ,str         ,str        , bool              , str                , 'datetime64[ns]', float    ,
              str                  , str                , str              , 'datetime64[ns]' , str          , str         , float            , str            , float         ,
              str                  , str                ,float             ,float             , float                           , float               , float      ]

    # Initialize the DataFrame with empty rows
    eval_df = pd.DataFrame(columns=columns)

    # Convert the column data types
    for col, dtype in zip(columns, dtypes):
        eval_df[col] = eval_df[col].astype(dtype)
        
    return eval_df

# %%
def first_date_N_years_ago(years):
    # Calculate start and end dates N years back
    end_date = datetime.date.today() - datetime.timedelta(days=1)
    start_date = end_date.replace(year=end_date.year - years, month=1, day=1)
    return start_date.strftime("%Y-%m-%d")


# %%
def load_sp500_list():
    stocks_list_csv = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp_df = pd.DataFrame()
    
    try:
        wiki_data=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies') # Open the link and download S&P company details in a table
        data = wiki_data[0] # All data is stored in first cell
        sp_df = data.sort_values(by=['Symbol'], ascending=True) # Sort the dataframe on ticker in alphabetical ascending order
    except:
        print("Cannot open file", stocks_list_csv,flush=True)

    # remove the dotted symbols, they are redundant 
    no_dot_symbols = [i for i in sp_df['Symbol'] if i.find('.')==-1]
    sp_df = sp_df[sp_df['Symbol'].isin(no_dot_symbols)]
    
    # returns count_row, count_col, df
    return sp_df.shape[0], sp_df.shape[1], sp_df


# %%
m,n,sp_df = load_sp500_list()
# sp_df[sp_df['Symbol'].isin([i for i in sp_df['Symbol'] if i.find('.')>0])]

# %%
def get_gstd(stock_data):
    # Calculate daily price changes
    price_changes = stock_data["Close"].pct_change()
    price_changes = price_changes.dropna()

    # Calculate the mean and geometric standard deviation of the daily price changes
    gstd = geo_std(price_changes)

    # Return the mean and geometric standard deviation
    return gstd

def get_sigma(stock_data):
    # Calculate daily price changes
    price_changes = stock_data["Close"].pct_change()

    mu, std = norm.fit(price_changes.dropna())
    return mu, std

# %%
def limit_trailing_stop_order_percentage(stock,stock_data,multiple_of_std_dev):
    close_price = stock_data.iloc[-1]['Close']
    mean, std_dev = get_sigma(stock_data)
    std_dev_pct = round(std_dev * 100,2)
    std_dev_dlr = round(close_price*std_dev,2)
    
    # use (std_dev_pct x multiple_of_std_dev) as the % to get the limit price
    
    pct_drop = -round(multiple_of_std_dev * std_dev_pct,2) 
    price_drop = round(close_price * pct_drop/100,2)
    min_trailing_stop_price = close_price + price_drop
    
    limit_price = min_trailing_stop_price + std_dev_dlr
    limit_price = round(limit_price,2)
    
    return close_price, pct_drop, price_drop, min_trailing_stop_price, limit_price, std_dev_pct, std_dev_dlr

# %%
def recommendation_table(eval_df,stock_list, lookback_years=1, sma_fast=40, sma_slow=200,
                         ema_trend_fast=20,ema_trend_slow=40, ema_trend_lookback=7,
                         run_update_models=False):
    stock_list = [str(s).upper() for s in stock_list]
    # print("Stock List:",stock_list)
    start_date = first_date_N_years_ago(lookback_years)
    today = datetime.datetime.today().strftime("%Y-%m-%d")
    if run_update_models :
        print(f"Updating Models (Lookback Period = {lookback_years} Years, Start Date = {start_date}) ...",end='',flush=True)
        stock_best_model_df = update_models(stock_list=stock_list)
        print("Done!",flush=True)

    print(f'Performing Analysis and Recommendations (Lookback Period = {lookback_years} Years, Start Date = {start_date}) ...',end='',flush=True)

    risk_free_rate = -1
    for s in stock_list:
        winner,buysell,buysell_date,close_price,stock_data, days_at_ST = run_supertrend(s,start_date)
        if(len(stock_data)<2):
            print(f"{s} data not found. Skipping!",end=",",flush=True)
            continue
        
        ema_last_date,ema_trend, df_trend_pdta, ema_buy_days = ema_trend_ind.calculate_pdta_alphatrend(stock=s,df=stock_data,Fast_EMA=ema_trend_fast,
                                                                                      Slow_EMA=ema_trend_slow,Lookback=ema_trend_lookback)
        df_trend_pdta = ema_lib.generate_signals(df_trend_pdta,fast_ema=ema_trend_fast,slow_ema=ema_trend_slow)
        df_trend_pdta = ema_trend_ind.get_momentum_indicators(df_trend_pdta)
        # print(df_trend_pdta[['Buy_Sell_Signal','Mom_Signal']])
        
        # Determine if Momentum is confirmed if both signals are 1
        last_momentum = df_trend_pdta['Mom_Signal'].iloc[-1]
        last_ema_trend = df_trend_pdta['Buy_Sell_Signal'].iloc[-1]
        
        if last_momentum == 1: 
            last_momentum_str = 'Buy'
        else: 
            last_momentum_str = 'Sell'

        consecutive_confirm_days = 0
        # if both signals are 1 then we have confirmation
        if last_ema_trend == 1 and last_momentum == 1:
            last_confirmation_str = 'Buy'
            # Carculate days count in confirmed momentum 
            last_date = df_trend_pdta.index[-1]
            
            # Find the number of days in 'Buy' recommendations
            for date in reversed(df_trend_pdta.index):
                # date_str = date.strftime('%Y-%m-%d')
                if df_trend_pdta.loc[date, 'Buy_Sell_Signal'] == 1 & df_trend_pdta.loc[date,'Mom_Signal'] == 1:
                    consecutive_confirm_days += 1
                else:
                    break            
        else:
            last_confirmation_str = 'Sell'
            consecutive_confirm_days = -1
        
        print(f"{s}",end=',',flush=True)
        mean, std_dev = get_sigma(stock_data)
        # std_dev = get_gstd(stock_data)
        std_dev_pct = round(std_dev * 100,2)
        std_dev_dlr = round(close_price*std_dev,2)
        sma_sig, sma_date, fastXslow = smacross.sma_xing(stock_data,sma_fast,sma_slow)
        ema_fast_slow_str = f"({ema_trend_fast}X{ema_trend_slow})"
        beta, market_data = sbeta.get_beta(stock_data)
        CAPM, VaR, Sharpe, risk_free_rate = capm.CAPM_VaR(stock_data=stock_data,market_data=market_data,bond_mat_duration = lookback_years,stock_beta=beta, risk_free_rate=risk_free_rate)
        Sharpe = round(Sharpe,2)
        LR_recommend = LR_run.get_recommendation(stock=s,lookback=lookback_years)
        LR_recommend_str = f"{LR_recommend[0]},{LR_recommend[1]},{LR_recommend[2]}"
        
        # columns = ['Last_Run'          ,'Stock'             ,'Last_Price'     ,'Std. Dev.%','$Std. Dev.','Supertrend_Winner' ,'Supertrend_Result','ST_Signal_Date', 'Days@ST',
        #            'LR_Best_Model'     ,'LR_Next_Day_Recomm', 'SMA_Crossed_Up','SMA_X_Date'             ,'SMA_FastXSlow'     ,  'EMA_Trend', 'EMA_Days_at_Buy','EMA_FastXSlow', 'EMA_Lookback',
        #            'Momentum'          , 'Confirmation'     ,'Mom_Days_Confirmed','Beta'         ,  'SharpeRatio'                  ,'CAPM'              ,'Daily_VaR'  ]
    
        if run_update_models:
            model_str = stock_best_model_df[stock_best_model_df['Stock'] == s].Model.values[0]
        else:
            model_str = 'N/A'
            
        new_row = [today           , s                  , close_price          ,f"+/-{std_dev_pct}%",f"+/-${std_dev_dlr}",    winner            , buysell         ,    buysell_date ,  days_at_ST,  
                   model_str       , LR_recommend_str   , sma_sig              , sma_date           , fastXslow          ,    ema_trend         , ema_buy_days    ,  ema_fast_slow_str,   ema_trend_lookback,
                   last_momentum_str   , last_confirmation_str  , consecutive_confirm_days, beta                 , Sharpe          , CAPM            , VaR  ]    
        
        eval_df = add_update(eval_df=eval_df,values = new_row)
        # print(eval_df[['Stock','EMA_Trend','Momentum','Confirmation','Mom_Days_Confirmed']])
        
    print("Done!",flush=True)
    return eval_df

# %%
def recommend_selling_strategy(lookback_years,stock_list,multiple_of_std_dev):
    """
    This function is used to evaluate the selling strategy of the stocks.
    It will return a DataFrame with the following columns:
    - Stock
    - $Last Price
    - Multiple of Std.Dev. Drop
    - Std. Dev.%
    - $Std. Dev.
    - %Trailing Stop Drop
    - $Trailing Stop Drop
    - $Min. Trailing Stop Price
    - $Recomm. Limit Price Order
    """

    print("Calculating Exit Strategies ....",end="", flush=True)
    # Initialize the DataFrame with empty rows
    start_date = first_date_N_years_ago(lookback_years)

    columns = ['Stock'          ,'$Last Price' , 'Multiple of Std Drop'    ,'Std. Dev.%','$Std. Dev.','%Trailing Stop Drop' , '$Trailing Stop Drop','$Min. Trailing Stop Price','$Recomm. Limit Price Order']
    dtypes = [str               , float        ,float                      ,float        , float     , float                , float                , float              , float]

    # Initialize the DataFrame with empty rows
    sell_orders_recomm = pd.DataFrame(columns=columns)

    # Convert the column data types
    for col, dtype in zip(columns, dtypes):
        sell_orders_recomm[col] = sell_orders_recomm[col].astype(dtype)


    for s in stock_list:
        winner,buysell,buysell_date,close_price,stock_data, days_at_ST = run_supertrend(s,start_date)
        if(len(stock_data)<2):
            print(f"{s.upper()} data not found. Skipping!",end=",",flush=True)
            continue
        else:
            close_price, pct_drop, price_drop, min_trailing_stop_price, limit_price, std_dev_pct, std_dev_dlr = \
                limit_trailing_stop_order_percentage(stock=s,stock_data=stock_data,multiple_of_std_dev=multiple_of_std_dev)
            # print(f"{s.upper()} Last Price: ${round(close_price,2)}, %Multiple of Std.Dev. Drop %:{multiple_of_std_dev}, % Trailing Drop:{pct_drop}%, \
            #       Limit Price:${limit_price}, Min Price:${round(min_trailing_stop_price,2)}, Std %:{std_dev_pct}, Std $:{std_dev_dlr}")
            
            new_row = [s.upper()  , f"${round(close_price,2)}"    , f"{multiple_of_std_dev}x"     ,f"+/-{std_dev_pct}%",f"+/-${std_dev_dlr}",    f"{pct_drop}%",    f"${price_drop}"   ,  f"${round(min_trailing_stop_price,2)}"  , f"${limit_price}"]    
            sell_orders_recomm = add_update(eval_df=sell_orders_recomm,values = new_row)
            
    print("....Done!",end="",flush=True)
    return sell_orders_recomm


# %%
def eval_all_sp500(lookback_years = 1,sma_fast = 50, sma_slow = 200, ema_trend_fast=20,ema_trend_slow=40, ema_trend_lookback=7,regenerate_models = False ):
    n,m,sp_df = load_sp500_list()
    stocks_sp500 = sp_df.Symbol.values
    stocks_sp500 = [s.strip().upper() for s in list(stocks_sp500)]

    eval_df = init_eval_table()
    eval_df = recommendation_table(eval_df,
                                stock_list=stocks_sp500, 
                                lookback_years=lookback_years, 
                                sma_fast=sma_fast, 
                                ema_trend_fast=ema_trend_fast,
                                ema_trend_slow=ema_trend_slow,
                                ema_trend_lookback=ema_trend_lookback,
                                sma_slow=sma_slow,run_update_models=regenerate_models)
    
    return eval_df

# %%
def eval_list_from_file(filename=os.path.dirname(__file__)+'/stocks_list.txt',lookback_years = 1,sma_fast = 50, sma_slow = 200 ,
                        ema_trend_fast=20,ema_days_at_buy=0,ema_trend_slow=40, ema_trend_lookback=7,
                        regenerate_models = False):
    stocks_from_file = read_stocklist(filename=filename)
    stocks_from_file = [s.strip().upper() for s in list(stocks_from_file)]
    today = datetime.datetime.today().strftime("%Y-%m-%d")

    eval_df = init_eval_table()
    eval_df = recommendation_table(eval_df,
                                stock_list=stocks_from_file, 
                                lookback_years=lookback_years, 
                                sma_fast=sma_fast, 
                                sma_slow=sma_slow,
                                ema_trend_fast=ema_trend_fast,
                                ema_trend_slow=ema_trend_slow,
                                ema_trend_lookback=ema_trend_lookback,
                                run_update_models=regenerate_models)
    
    return eval_df

# %%
def screen_for_buys(eval_df, ignore_supertrend_winners=False):
        if not ignore_supertrend_winners:
                buys_df = eval_df[ (eval_df['Supertrend_Winner']==True) &  
                        (eval_df['Supertrend_Result']=='Buy') & 
                        (eval_df['LR_Next_Day_Recomm'] == 'Buy,Buy,Buy') &
                        (eval_df['SMA_Crossed_Up']=='Buy')].sort_values(by=['Supertrend_Winner','Supertrend_Result',
                                                                            'ST_Signal_Date','SMA_Crossed_Up','SMA_X_Date'],
                                                                        ascending=[False,True,False,True,False])
        else:
                buys_df = eval_df[ (eval_df['Supertrend_Result']=='Buy') & 
                        (eval_df['LR_Next_Day_Recomm'] == 'Buy,Buy,Buy') &
                        (eval_df['SMA_Crossed_Up']=='Buy')].sort_values(by=['Supertrend_Winner','Supertrend_Result',
                                                                            'ST_Signal_Date','SMA_Crossed_Up','SMA_X_Date'],
                                                                        ascending=[False,True,False,True,False])            
        
        return buys_df

# %%
def screen_for_sells(eval_df, ignore_supertrend_winners=False):
        if not ignore_supertrend_winners:
                sells_df = eval_df[ (eval_df['Supertrend_Winner']==True) &  
                        ((eval_df['Supertrend_Result']=='Sell') |
                        (eval_df['LR_Next_Day_Recomm'] == 'Sell,Sell,Sell') |
                        (eval_df['SMA_Crossed_Up']=='Sell')) ].sort_values(by=['Supertrend_Winner','Supertrend_Result',
                                                                            'ST_Signal_Date','SMA_Crossed_Up','SMA_X_Date'],
                                                                        ascending=[False,True,False,True,False])
        else:
                sells_df = eval_df[ (eval_df['Supertrend_Result']=='Sell') |
                        (eval_df['LR_Next_Day_Recomm'] == 'Sell,Sell,Sell') |
                        (eval_df['SMA_Crossed_Up']=='Sell')].sort_values(by=['Supertrend_Winner','Supertrend_Result',
                                                                            'ST_Signal_Date','SMA_Crossed_Up','SMA_X_Date'],
                                                                        ascending=[False,True,False,True,False])            
        
        return sells_df


def send_email(recipient, subject, body, html=False):
    print(f"Sending EMAIL to {recipient}",end='',flush=True)
    # Construct the command to send the email
    if html:
        # Specify content type as HTML
        message = f"{body}"
        # Write the message to a temporary file
        with open("/tmp/stocks_email.txt", "w") as f:
            f.write(message)

        # Use Mutt to send the email with the temporary file
        command = f"mutt  -e 'set content_type=text/html' -s '{subject}' {recipient}  < /tmp/stocks_email.txt"
        # print(message)
        # print("Content type = HTML",flush=True)
    else:
        # Default to plain text content
        command = f'echo "{body}" | mutt -s "{subject}" {recipient}'
        # print(body,flush=True)
        # print("Content type = TEXT",flush=True)
        # Execute the command
        
    # print(command,flush=True)
    os.system(command)
    print("...Done!",flush=True)


# Example usage
recipient_email = "sabawi@gmail.com"
email_subject = "Stocks Evaluation Completed"



# %%
results_dir = os.getcwd()+'/eval_results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
fname = f'{results_dir}/Eval_Results_{datetime.datetime.today().strftime("%Y_%m_%dT%I%M%S%p")}.csv'

# %% [markdown]
# ### Evaluate Stocks List

# %%

# Regenrate the models every Friday only
regenerate_models = True  if datetime.datetime.today().weekday() == 4 else False
# symbols_file = os.path.dirname(__file__)+'/stocks_portfolio.txt'
# symbols_file = os.path.dirname(__file__)+'/stocks_list.txt'
symbols_file = os.path.dirname(__file__)+'/stocks_list5.txt' ## Inclusive of NASDAQ 100
# symbols_file = os.path.dirname(__file__)+'/stocks_list3.txt'
# symbols_file = os.path.dirname(__file__)+'/sectors_etfs.txt'
stocks_list = read_stocklist(symbols_file)  

lookback_years = 2

eval_df = eval_list_from_file(filename=symbols_file,lookback_years=lookback_years,sma_fast=50,sma_slow=100, 
                              ema_trend_fast=20, ema_trend_slow=40, ema_trend_lookback=7,regenerate_models=regenerate_models)
# eval_df_from_file = eval_df
# eval_df = eval_all_sp500(lookback_years=2,sma_fast=50,sma_slow=200, regenerate_models=regenerate_models)
## write_to_csv(eval_df)
eval_df.to_csv(fname)
insert_data_into_database(eval_df)
insert_data_into_database_old(eval_df)

# %% [markdown]
# ### All Results

# %%
# print(f"{len(eval_df)} Stocks:",flush=True)
# display(HTML(eval_df.sort_values(['ST_Signal_Date','SMA_X_Date'], ascending=False).to_html(index=False)))
ret = df_to_html_file(eval_df,"/var/www/html/home/viewable_pages/stock_evaluation.php","Latest Stock Evaluation: All") 

# %% [markdown]
# ### Exit Strategy

# %%
# lookback_years = 2
# my_stock_list = ['amd','jbl','aapl','nvda','msft','googl','amzn','nflx','meta','adbe','cdns','de','avgo','orcl']
my_stock_list = stocks_list
multiple_of_std_dev = 1.5 # Stop Loss as percentage of Std.Dev.
sell_orders_recomm = recommend_selling_strategy(lookback_years,my_stock_list,multiple_of_std_dev)

# print("Order Type: Limit Trailing Stop Loss Percent (%) Orders ONLY:  ",flush=True)
# print("=============================================================",flush=True)
# print(f"{len(sell_orders_recomm)} Stocks:",flush=True)
# display(HTML(sell_orders_recomm.to_html(index=False)))
ret = df_to_html_file(sell_orders_recomm,"/var/www/html/home/viewable_pages/exit_strategy.php","Exit Strategies Recommended <br>Order Type: Limit Trailing Stop Loss Percent (%) Orders ONLY") 

# %% [markdown]
# ### Send to Printer

# %%
if symbols_file == os.path.dirname(__file__)+'/stocks_list.txt':
    eval_df.to_csv(os.path.dirname(__file__)+"/stocks_eval_df_table.csv")
elif symbols_file == os.path.dirname(__file__)+'/sectors_etfs.txt':
    eval_df.to_csv(os.path.dirname(__file__)+"/sectors_etfs_eval_df_table.csv")
    
print_it = False

def send_df_to_printer(df):
    pdf_filename = "full_eval_df_table.pdf"

    fig, ax =plt.subplots(figsize=(12,4))
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=df.values,colLabels=df.columns,loc='top')

    pp = PdfPages(pdf_filename)
    pp.savefig(fig, bbox_inches='tight')
    pp.close()

    os.system("lpr "+pdf_filename)
    
if (print_it):
    send_df_to_printer(eval_df)

# %% [markdown]
# ### Buy, Buy, and more Buys

# %%
import time
time.sleep(0.5)
buys_eval_df= screen_for_buys(eval_df=eval_df,ignore_supertrend_winners=False)
buys_eval_df = buys_eval_df.sort_values('Daily_VaR',ascending=False).sort_values('Sharpe_Ratio%',ascending=True).sort_values(by='SMA_X_Date', ascending=False)
# print(f"{len(buys_eval_df)} Stocks:",flush=True)
sublist = ','.join(buys_eval_df['Stock'].astype(str))
# print(sublist,flush=True)
save_list_to_file(sublist,os.getcwd()+'/picked_stocks','BuyBuyBuy.csv')
# display(HTML(buys_eval_df.to_html(index=False)))
ret = df_to_html_file(buys_eval_df,"/var/www/html/home/viewable_pages/buybuybuy.php","Buy, Buy, Buy Rated Stocks") 

# START AN EMAIL
email_body =  " Run is Done! \n"+buys_eval_df.to_string()
page_header = ''' 
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0"> </head>'''
        

# Start of html body
email_body_html = page_header + "\n<body>"
# ########################################


# ADD TO EMAIL BODY
# ########################################
# BuyBuyBuy tables 
email_body_html = email_body_html + "<h2>Buy, Buy, and More Buy</h2>"
email_body_html = email_body_html + buys_eval_df.to_html(border=True,index=False)


# Make a comma separated list of stocks in one line
# linktext = "https://finviz.com/quote.ashx?t=????&p=d"
linktext = "https://digital.fidelity.com/prgw/digital/research/quote/dashboard/summary?symbol=????"
buys_eval_df1 = buys_eval_df.copy()
buys_eval_df1 = make_text_clickable(buys_eval_df1,'Stock',linktext,'????')

stock_list = ', '.join(buys_eval_df1['Stock'].astype(str))
email_body_html = email_body_html + stock_list
# ##########################################


# SEND EMAIL NOW
# End html body
email_body_html = email_body_html +"</body>\n" + "</html>"

email_subject = "Stocks: Buy, Buy, and More Buy"

# email_body_html = buys_eval_df.to_html()
send_email(recipient=recipient_email, subject=email_subject, body=email_body_html, html=True )

# %% [markdown]
# ### Buy, Buy, and more Buys (Not necessarily ST winners)

# %%
time.sleep(0.5)
buys_eval_df2= screen_for_buys(eval_df=eval_df,ignore_supertrend_winners=True)
buys_eval_df2 = buys_eval_df2.sort_values(by='SMA_X_Date', ascending=False).sort_values('Sharpe_Ratio%',ascending=True).sort_values('Daily_VaR',ascending=False)
# print(f"{len(buys_eval_df2)} Stocks:",flush=True)
# print(','.join(buys_eval_df2['Stock'].astype(str)),flush=True)
# display(HTML(buys_eval_df2.to_html(index=False)))
ret = df_to_html_file(buys_eval_df2,"/var/www/html/home/viewable_pages/buybuybuy2.php","Buy, Buy, Buy Rated Stocks (Not necessarily ST winners)") 

# %% [markdown]
# ### Save the Buys

# %%
eval_df.to_csv(os.path.dirname(__file__)+"/stocks_buys_eval_df_table.csv")

# %% [markdown]
# ### Time to Sell!

# %%
sells_eval_df= screen_for_sells(eval_df=eval_df,ignore_supertrend_winners=True)
# print(f"{len(sells_eval_df)} Stocks:",flush=True)
# print(','.join(sells_eval_df['Stock'].astype(str)),flush=True)
# display(HTML(sells_eval_df.to_html(index=False)))
ret = df_to_html_file(sells_eval_df,"/var/www/html/home/viewable_pages/time_to_sell.php","Time to Sell") 

# %% [markdown]
# ### All Roads Lead to UP & Safe

# %%
time.sleep(0.5)
buys_safe = buys_eval_df[(buys_eval_df['Beta']>0.8) & (buys_eval_df['Beta']<2) ].sort_values(by='SMA_X_Date', ascending=False).sort_values('Daily_VaR',ascending=False)
# print(f"{len(buys_safe)} Stocks:",flush=True)
# print(','.join(buys_safe['Stock'].astype(str)),flush=True)
# display(HTML(buys_safe.to_html(index=False)))
ret = df_to_html_file(buys_safe,"/var/www/html/home/viewable_pages/all_roads_lead_to_up_safe.php","All Roads Lead to UP & Safe") 
save_list_to_file(sublist,os.getcwd()+'/picked_stocks','all_up_and_safe.csv')

# START AN EMAIL
email_body =  " Run is Done! \n"+buys_safe.to_string()
page_header = ''' 
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0"> </head>'''
        

# Start of html body
email_body_html = page_header + "\n<body>"
# ########################################


# ADD TO EMAIL BODY
# ########################################
# BuyBuyBuy tables 
email_body_html = email_body_html + "<h2>All Roads Lead to UP & Safe</h2>"
email_body_html = email_body_html + buys_safe.to_html(border=True,index=False)


# Make a comma separated list of stocks in one line
# linktext = "https://finviz.com/quote.ashx?t=????&p=d"
linktext = "https://digital.fidelity.com/prgw/digital/research/quote/dashboard/summary?symbol=????"
buys_safe1 = buys_safe.copy()
buys_safe1 = make_text_clickable(buys_safe1,'Stock',linktext,'????')

stock_list = ', '.join(buys_safe1['Stock'].astype(str))
email_body_html = email_body_html + stock_list
# ##########################################


# SEND EMAIL NOW
# End html body
email_body_html = email_body_html +"</body>\n" + "</html>"

email_subject = "Stocks: All Roads Lead to UP & Safe"

# email_body_html = buys_eval_df.to_html()
send_email(recipient=recipient_email, subject=email_subject, body=email_body_html, html=True )


# %% [markdown]
# ### Save the Safe-Buys as Top Picks

# %%
buys_safe.to_csv(os.path.dirname(__file__)+"/top_picks.csv")

# %% [markdown]
# ### High Sharpe Ratios (Top 15)

# %%
# hi_sharpe_df = eval_df.sort_values(['Sharpe_Ratio%'], ascending=False)
hi_sharpe_df = eval_df.sort_values(by=['Sharpe_Ratio%', 'CAPM', 'Momentum', 'Last_Price', 'Std._Dev.%', 'Supertrend_Result', 'SMA_Crossed_Up', 'EMA_Trend', 'Beta', 'Daily_VaR'], 
                           ascending=[False, False, False, False, True, False, False, False, False, True])

display(HTML(hi_sharpe_df.head(15).to_html(index=False)))
ret = df_to_html_file(hi_sharpe_df,"/var/www/html/home/viewable_pages/high_sharpe_ratios.php","High Sharpe Ratios:<br>The Sharpe ratio compares the return of an investment with its risk. Generally, the higher the more attractive the risk-adjusted return.") 

# %% [markdown]
# ### UP the Next Day

# %%
LR_Next_Day_Recomm_only = eval_df[eval_df['LR_Next_Day_Recomm']=='Buy,Buy,Buy'].sort_values('Daily_VaR',ascending=False)
# print(f"{len(LR_Next_Day_Recomm_only)} Stocks:",flush=True)
# print(','.join(LR_Next_Day_Recomm_only['Stock'].astype(str)),flush=True)
# display(HTML(LR_Next_Day_Recomm_only.to_html(index=False)))
ret = df_to_html_file(LR_Next_Day_Recomm_only,"/var/www/html/home/viewable_pages/up_next_day.php","Linear Reg. Predicted Up_Next_Day_only") 

# %% [markdown]
# ### Down the Next Day

# %%
LR_Next_Day_Sell_only = eval_df[(eval_df['LR_Next_Day_Recomm']=='Sell,Buy,Buy') | 
                                (eval_df['LR_Next_Day_Recomm']=='Sell,Sell,Sell') | 
                                (eval_df['LR_Next_Day_Recomm']=='Sell,Buy,Sell') | 
                                (eval_df['LR_Next_Day_Recomm']=='Buy,Sell,Sell')]

# print(f"{len(LR_Next_Day_Sell_only)} Stocks:",flush=True)
# print(','.join(LR_Next_Day_Sell_only['Stock'].astype(str)),flush=True)
# display(HTML(LR_Next_Day_Sell_only.to_html(index=False)))
ret = df_to_html_file(LR_Next_Day_Sell_only,"/var/www/html/home/viewable_pages/down_next_day.php","Linear Reg. Predicted Down_Next_Day_only") 

# %% [markdown]
# ### Supertrend Winners and Still Supertrending

# %%
up_Supertrend = eval_df[(eval_df['Supertrend_Result'] == 'Buy') & (eval_df['Supertrend_Winner'] == True)].sort_values('ST_Signal_Date', ascending=False).sort_values(by='SMA_X_Date', ascending=False)
# print(f"{len(up_Supertrend)} Stocks:",flush=True)
# display(HTML(up_Supertrend.to_html(index=False)))
ret = df_to_html_file(up_Supertrend,"/var/www/html/home/viewable_pages/up_supertrend.php","Supertrend Winners and Still Supertrending") 

# %% [markdown]
# ### Supertrending (Winners  or Not)

# %%
up_Supertrend = eval_df[eval_df['Supertrend_Result'] == 'Buy'].sort_values('ST_Signal_Date', ascending=False).sort_values(by='SMA_X_Date', ascending=False)
# print(f"{len(up_Supertrend)} Stocks:",flush=True)
# display(HTML(up_Supertrend.to_html(index=False)))
ret = df_to_html_file(up_Supertrend,"/var/www/html/home/viewable_pages/up_supertrend_not_winners.php","Supertrending (Winners  or Not)") 


# %% [markdown]
# ### Fast SMA Crossed Slow SMA

# %%
Crossed_up = eval_df[eval_df['SMA_Crossed_Up'] == 'Buy'].sort_values('SMA_X_Date', ascending=False)
# print(f"{len(Crossed_up)} Stocks",flush=True)
# display(HTML(Crossed_up.to_html(index=False)))
ret = df_to_html_file(Crossed_up,"/var/www/html/home/viewable_pages/crossed_up.php","Fast SMA Crossed Slow SMA") 

# %% [markdown]
# ### SMA Crossed and in Supertrend & Winner

# %%
Crossed_up = eval_df[(eval_df['SMA_Crossed_Up'] == 'Buy') & (eval_df['Supertrend_Result'] == 'Buy') & (eval_df['Supertrend_Winner'] == True) ].sort_values('SMA_X_Date', ascending=False)
sublist2 = ','.join(Crossed_up['Stock'].astype(str))
# print(sublist2,flush=True)
save_list_to_file(sublist2,'picked_stocks','SMAX_Supertrend_Winner.csv')
# print(f"{len(Crossed_up)} Stocks:",flush=True)
# display(Crossed_up.hide_index())
# display(HTML(Crossed_up.to_html(index=False)))
ret = df_to_html_file(Crossed_up,"/var/www/html/home/viewable_pages/crossed_up_supertrend.php","SMA Crossed and in Supertrend & Winner") 

# %% [markdown]
# ### Test One or a Group of Stocks

# %%
"""
stock_to_test = 'swks'
single_stock_df = init_eval_table()
single_stock_list = [stock_to_test]
single_stock_list = ['jbl','aapl','nvda','msft','googl','amzn','nflx','meta','adbe','cdns','de','avgo','orcl']
single_stock_df = recommendation_table(single_stock_df,single_stock_list, lookback_years=5, sma_fast=50, sma_slow=200,run_update_models=False)
display(HTML(single_stock_df.sort_values('SMA_X_Date', ascending=False).to_html(index=False)))

# %% [markdown]
# ### Filter the Group

# %%
single_stock_df= screen_for_buys(eval_df=single_stock_df,ignore_supertrend_winners=True)
display(HTML(single_stock_df.sort_values('SMA_X_Date', ascending=False).to_html(index=False)))
"""

# %%
import requests
import textwrap
nasdaq_api_key = 'yYCEL8BqzxYUsgG67FTb'

def get_company_info(ticker_symbol):
    ticker_symbol = str(ticker_symbol).upper()
    # Set the endpoint URL and the API key
    url = "https://api.nasdaq.com/api/company/{}/company-profile"
    api_key = nasdaq_api_key

    # Format the endpoint URL with the ticker symbol
    endpoint_url = url.format(ticker_symbol)

    # Set the headers and parameters for the request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 Edge/16.16299",
        "Accept-Language": "en-US,en;q=0.9"
    }

    params = {
        "apikey": nasdaq_api_key
    }

    # Send the request and get the response
    response = requests.get(endpoint_url, headers=headers, params=params)

    # Parse the response JSON
    response_json = response.json()

    # Get the company full name and sector
    if(response_json["status"]["rCode"] == 200):
        dic = {
        "Ticker" : ticker_symbol,
        "Full_Name" : response_json["data"]["CompanyName"]["value"],
        "Sector" : response_json["data"]["Sector"]["value"],
        "Industry" : response_json["data"]["Industry"]["value"],
        "Region" : response_json["data"]["Region"]["value"],
        "Address" : response_json["data"]["Address"]["value"],
        "Description" : response_json["data"]["CompanyDescription"]["value"]
        }    
        df = pd.DataFrame.from_dict(dic, orient='index').T
        return df
    else:
        print("Return Code : ",response_json["status"]["rCode"],flush=True)
        print("Error: Symbol Does not exist",flush=True)
        return pd.DataFrame()
    
"""
company_info = get_company_info('ivr')
if(not company_info.empty):
    print(f"Company  : {company_info.loc[0]['Full_Name']}")
    print(f"Sector   : {company_info.loc[0]['Sector']}")
    print(f"Industry : {company_info.loc[0]['Industry']}")
    print(f"Region   : {company_info.loc[0]['Region']}")

    wrapped_text = textwrap.wrap(company_info.loc[0]['Description'],100)
    for line in wrapped_text:
        print(line)

"""

# %%
import os
import json

def build_sp500_companies_database():
    n, m, sp_df = load_sp500_list()
    stocks_sp500 = sp_df.Symbol.values
    stocks_sp500 = [s.strip().upper() for s in list(stocks_sp500)]
    
    if not os.path.exists(os.path.dirname(__file__)+"/company_info"):
        os.makedirs(os.path.dirname(__file__)+"/company_info")
    
    for s in stocks_sp500:
        company_info = get_company_info(s)
        sfile = f"/company_info/{s}.json"
        filename = os.path.dirname(__file__)+sfile
        print(f"Writing file {filename}",flush=True)
        with open(filename, "w") as outfile:
            json.dump(company_info, outfile, indent=4)
        
        print(f"Saved {filename}",flush=True)

    # print("Done!")
    return sp_df

# sp_df = build_sp500_companies_database()

# %%

# Uncomment the next 3 lines to regenerate the list of SP 500 companies from wiki source
# n, m, sp_df = load_sp500_list()
# sp_df = sp_df.rename(columns = {'Symbol':'Stock'})
# ret = df_to_html_file(sp_df,"/var/www/html/home/viewable_pages/sp500_companies_info.php","SP500 Companies Information") 
# sp_df.tail(50)

print("Run Completed")
