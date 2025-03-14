
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from math import floor
from termcolor import colored as cl
from datetime import datetime
import yfinance as yf

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)

import warnings
from functools import wraps

def suppress_warnings(func):
  """Decorator to suppress warnings within a function."""
  @wraps(func)
  def wrapper(*args, **kwargs):
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=FutureWarning)
      return func(*args, **kwargs)
  return wrapper

# # EXTRACTING DATA

# In[395]:

def get_historical_data(symbol, start_date):

    df = yf.Ticker(symbol).history(interval="1d", start=start_date, end=datetime.now())
    return df


from datetime import datetime, timedelta

def get_time_difference(past_date):
    # Convert the past date string to a datetime object
    past_date = datetime.strptime(past_date, '%Y-%m-%d')

    # Get the current date
    current_date = datetime.now()

    # Calculate the difference between the current date and the past date
    time_difference = current_date - past_date

    # Calculate the number of years, months, weeks, and days
    years = time_difference.days // 365
    months = time_difference.days // 30
    weeks = time_difference.days // 7
    days = time_difference.days

    return years, months, weeks, days


@suppress_warnings
def get_supertrend(high, low, close, lookback, multiplier):
    
    # ATR
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
    atr = tr.ewm(lookback).mean()
    
    # H/L AVG AND BASIC UPPER & LOWER BAND
    
    hl_avg = (high + low) / 2
    upper_band = (hl_avg + multiplier * atr).dropna()
    lower_band = (hl_avg - multiplier * atr).dropna()
    
    # FINAL UPPER BAND
    
    final_bands = pd.DataFrame(columns = ['upper', 'lower'])
    final_bands.iloc[:,0] = [x for x in upper_band - upper_band]
    # final_bands.iloc[:,1] = final_bands.iloc[:,0]
    
    final_bands[final_bands.columns[1:2]] = final_bands.iloc[:, 0:1].values
    
    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i,0] = 0
        else:
            if (upper_band[i] < final_bands.iloc[i-1,0]) | (close[i-1] > final_bands.iloc[i-1,0]):
                final_bands.iloc[i,0] = upper_band[i]
            else:
                final_bands.iloc[i,0] = final_bands.iloc[i-1,0]
    
    # FINAL LOWER BAND
    
    for i in range(len(final_bands)):
        if i == 0:
            final_bands.iloc[i, 1] = 0
        else:
            if (lower_band[i] > final_bands.iloc[i-1,1]) | (close[i-1] < final_bands.iloc[i-1,1]):
                final_bands.iloc[i,1] = lower_band[i]
            else:
                final_bands.iloc[i,1] = final_bands.iloc[i-1,1]
    
    # SUPERTREND
    
    supertrend = pd.DataFrame(columns = [f'supertrend_{lookback}'])
    supertrend.iloc[:,0] = [x for x in final_bands['upper'] - final_bands['upper']]
    
    for i in range(len(supertrend)):
        if i == 0:
            supertrend.iloc[i, 0] = 0
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 0] and close[i] < final_bands.iloc[i, 0]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 0] and close[i] > final_bands.iloc[i, 0]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 1] and close[i] > final_bands.iloc[i, 1]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
        elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 1] and close[i] < final_bands.iloc[i, 1]:
            supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
    
    supertrend = supertrend.set_index(upper_band.index)
    supertrend = supertrend.dropna()[1:]
    
    # ST UPTREND/DOWNTREND
    
    upt = []
    dt = []
    close = close.iloc[len(close) - len(supertrend):]

    for i in range(len(supertrend)):
        if close[i] > supertrend.iloc[i, 0]:
            upt.append(supertrend.iloc[i, 0])
            dt.append(np.nan)
        elif close[i] < supertrend.iloc[i, 0]:
            upt.append(np.nan)
            dt.append(supertrend.iloc[i, 0])
        else:
            upt.append(np.nan)
            dt.append(np.nan)
            
    st, upt, dt = pd.Series(supertrend.iloc[:, 0]), pd.Series(upt), pd.Series(dt)
    upt.index, dt.index = supertrend.index, supertrend.index
    
    return st, upt, dt


@suppress_warnings
def implement_st_strategy(prices, st):
    buy_price = []
    sell_price = []
    st_signal = []
    signal = 0
    
    for i in range(len(st)):
        if st[i-1] > prices[i-1] and st[i] < prices[i]:
            if signal != 1:
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1
                st_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                st_signal.append(0)
        elif st[i-1] < prices[i-1] and st[i] > prices[i]:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(prices[i])
                signal = -1
                st_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                st_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            st_signal.append(0)
            
    return buy_price, sell_price, st_signal


# # START ANALYSIS
@suppress_warnings
def supertrend(stock,start_date):
    stock_data = get_historical_data(stock, start_date).tz_localize(None)
    if(len(stock_data)<2):
        print(f"{stock} data not found. Skipping",end=",")
        return None, None, None, None, stock_data
    
    # print(stock_data)
    Lookback = 15 # in days
    Multiplier = 3
    stock_data['st'], stock_data['s_upt'], stock_data['st_dt'] = \
        get_supertrend(stock_data['High'], stock_data['Low'], stock_data['Close'], Lookback, Multiplier)
    stock_data = stock_data[1:]
    # print(stock_data.head())

    buy_price, sell_price, st_signal = implement_st_strategy(stock_data['Close'], stock_data['st'])

    position = []
    for i in range(len(st_signal)):
        if st_signal[i] > 1:
            position.append(0)
        else:
            position.append(1)
            
    for i in range(len(stock_data['Close'])):
        if st_signal[i] == 1:
            position[i] = 1
        elif st_signal[i] == -1:
            position[i] = 0
        else:
            position[i] = position[i-1]
            
    close_price = stock_data['Close']
    st = stock_data['st']
    st_signal = pd.DataFrame(st_signal).rename(columns = {0:'st_signal'}).set_index(stock_data.index)
    position = pd.DataFrame(position).rename(columns = {0:'st_position'}).set_index(stock_data.index)

    frames = [close_price, st, st_signal, position]
    strategy = pd.concat(frames, join = 'inner', axis = 1)

    only_buysell_signals_df = pd.DataFrame(columns=["Date","Buy_Sell","Close_Price"])
    
    # print(strategy)
    # for index, row in st_signal.iterrows():
    for index, row in strategy[strategy['st_signal'] != 0].iterrows():
        if row['st_signal'] != 0:
            buysell = None
            # print(index, row['st_signal'])
            if row['st_signal']==1:
                buysell = "Buy"
            elif row['st_signal'] == -1:
                buysell = 'Sell'

            dic = {"Date": [index.strftime("%Y-%m-%d")], "Buy_Sell": [buysell], "Close_Price": [row['Close']]}
            tmp_df = pd.DataFrame(dic)
            # only_buysell_signals_df=pd.concat([only_buysell_signals_df,tmp_df])
            only_buysell_signals_df = pd.concat([only_buysell_signals_df,tmp_df], ignore_index=True, sort=False)
    only_buysell_signals_df.set_index("Date",inplace=True)
    past_date = only_buysell_signals_df.index[-1]
    styears, stmonths, stweeks, stdays = get_time_difference(past_date)


    if only_buysell_signals_df.iloc[-1].Buy_Sell == 'Buy':
        last_signal = 'Buy'
    else:
        last_signal = 'Sell'
    last_signal_date = only_buysell_signals_df.index[-1]
    
    stock_ret = pd.DataFrame(np.diff(stock_data['Close'])).rename(columns = {0:'returns'})
    st_strategy_ret = []

    for i in range(len(stock_ret)):
        returns = stock_ret['returns'][i]*strategy['st_position'].iloc[i]
        st_strategy_ret.append(returns)
        
    st_strategy_ret_df = pd.DataFrame(st_strategy_ret).rename(columns = {0:'st_returns'})
    investment_value = 100000
    number_of_stocks = floor(investment_value/stock_data['Close'].iloc[-1])
    st_investment_ret = []

    for i in range(len(st_strategy_ret_df['st_returns'])):
        returns = number_of_stocks*st_strategy_ret_df['st_returns'].iloc[i]
        st_investment_ret.append(returns)

    st_investment_ret_df = pd.DataFrame(st_investment_ret).rename(columns = {0:'investment_returns'})
    total_investment_ret = round(sum(st_investment_ret_df['investment_returns']), 2)
    profit_percentage = floor((total_investment_ret/investment_value)*100)
    def get_benchmark(start_date, investment_value):
        spy = get_historical_data('SPY', start_date)['Close']
        benchmark = pd.DataFrame(np.diff(spy)).rename(columns = {0:'benchmark_returns'})
        
        investment_value = investment_value
        number_of_stocks = floor(investment_value/spy.iloc[-1])
        benchmark_investment_ret = []
        
        for i in range(len(benchmark['benchmark_returns'])):
            returns = number_of_stocks*benchmark['benchmark_returns'].iloc[i]
            benchmark_investment_ret.append(returns)

        benchmark_investment_ret_df = pd.DataFrame(benchmark_investment_ret).rename(columns = {0:'investment_returns'})
        return benchmark_investment_ret_df

    benchmark = get_benchmark(start_date, 100000)
    investment_value = 100000
    total_benchmark_investment_ret = round(sum(benchmark['investment_returns']), 2)
    benchmark_profit_percentage = floor((total_benchmark_investment_ret/investment_value)*100)
    winner_strategy = False
    if(profit_percentage - benchmark_profit_percentage > 0.0):
        winner_strategy = True
        # print(cl('                         WINNING STRATEGY                        ', attrs=['bold']))
        from pathlib import Path
        wpath = Path('./ST_winners.csv')
        file_exists = wpath.is_file()
        with open(wpath, "a") as myfile:
            if not file_exists:
                myfile.write("Date_Checked,stock,Percent_Above_SPY\n")
            myfile.write(f"{datetime.today().strftime('%Y-%m-%d')},{stock.upper()},{str(profit_percentage - benchmark_profit_percentage)}\n")

    return winner_strategy, last_signal, last_signal_date, round(stock_data.iloc[-1].Close,2), stock_data, stdays

if __name__ == "__main__":
    w,s,d,close,stock_data,stdays = supertrend('AAPL','2020-01-01')
    print(f"Winner:{w}\nLast Signal:{s},\nLast Signal Date:{d},\nStock Data:{close},\nDays in Supertrend:{stdays}")



