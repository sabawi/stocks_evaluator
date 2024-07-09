import sqlite3
import pymysql
import csv

# Step 1: Export SQLite data to CSV
def export_sqlite_to_csv():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM stock_data')
    
    with open('stock_data.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write header
        csv_writer.writerow([i[0] for i in cursor.description])
        # Write data
        csv_writer.writerows(cursor)
    
    conn.close()

export_sqlite_to_csv()

# Step 2: Import CSV data into MySQL
def import_csv_to_mysql():
    mysql_conn = pymysql.connect(
        host='localhost',
        user='root',
        password='Down2earth!',
        database='mystocks',
        local_infile=True
    )
    mysql_cursor = mysql_conn.cursor()

    mysql_cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_data (
            `Last Run` VARCHAR(255),
            `Stock` VARCHAR(255),
            `Last Price` DOUBLE,
            `%Std. Dev.` VARCHAR(255),
            `$Std. Dev.` VARCHAR(255),
            `Supertrend Winner` TINYINT(1),
            `Supertrend Result` VARCHAR(255),
            `ST Signal_Date` VARCHAR(255),
            `Days@ST` DOUBLE,
            `LR Best_Model` VARCHAR(255),
            `LR Next_Day Recomm` VARCHAR(255),
            `SMA Crossed_Up` VARCHAR(255),
            `SMA_X_Date` VARCHAR(255),
            `SMA FastXSlow` VARCHAR(255),
            `Beta` DOUBLE,
            `%Sharpe Ratio` DOUBLE,
            `CAPM` DOUBLE,
            `Daily VaR` DOUBLE,
            `EMA_Trend` VARCHAR(255),
            `EMA_FasSlow` VARCHAR(255),
            `EMA_Lookback` INT,
            PRIMARY KEY(`Last Run`, `Stock`)
        );
    ''')

    # Load data into MySQL
    mysql_cursor.execute('''
        LOAD DATA LOCAL INFILE 'stock_data.csv'
        INTO TABLE stock_data
        FIELDS TERMINATED BY ','
        ENCLOSED BY '"'
        LINES TERMINATED BY '\n'
        IGNORE 1 ROWS;
    ''')

    mysql_conn.commit()
    mysql_conn.close()

import_csv_to_mysql()

