#!/usr/bin/env python
# coding: utf-8

# In[203]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20,10)


# In[204]:


def generate_signals(data,fast_sma=50,slow_sma=200):
    data[f'{fast_sma}_day_SMA'] = data['Close'].rolling(window=fast_sma).mean()
    data[f'{slow_sma}_day_SMA'] = data['Close'].rolling(window=slow_sma).mean()
    data['signal'] = 0.0

    for i in range(len(data) - 1):
        if data[f'{fast_sma}_day_SMA'][i+1] > data[f'{slow_sma}_day_SMA'][i+1] and data[f'{fast_sma}_day_SMA'][i] <= data[f'{slow_sma}_day_SMA'][i]:
            data.loc[data.index[i+1], 'signal'] = 1
        elif data[f'{fast_sma}_day_SMA'][i+1] < data[f'{slow_sma}_day_SMA'][i+1] and data[f'{fast_sma}_day_SMA'][i] >= data[f'{slow_sma}_day_SMA'][i]:
            data.loc[data.index[i+1], 'signal'] = -1

    return data


# In[205]:


def implement_signal_strategy(prices, st):
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


# In[206]:

def sma_xing(data:pd.DataFrame,fast_sma:int, slow_sma:int):
    # stock = str(input("Enter Stock Symbol : ")).upper()
    # fast_sma = int(input("Enter the Fast Simple Moving Average Period : "))
    # slow_sma = int(input("Enter the Slow Simple Moving Average Period : "))
    # # Download stock daily data
    # data = yf.download(stock, start='2010-01-01', end='2023-03-17', interval='1d')

    # Generate buy and sell signals
    stock_data = generate_signals(data,fast_sma=fast_sma,slow_sma=slow_sma)
    buy_price, sell_price, st_signal = implement_signal_strategy(stock_data['Close'], stock_data['signal'])


    stock_data_noindex = stock_data.reset_index().copy()
    buy_price_list = [(stock_data_noindex.iloc[i]['Date'],stock_data_noindex.iloc[i]['Close'])  for i in range(len(stock_data_noindex)) if stock_data_noindex.iloc[i].signal == 1 ]
    sell_price_list = [(stock_data_noindex.iloc[i]['Date'],stock_data_noindex.iloc[i]['Close'])  for i in range(len(stock_data_noindex)) if stock_data_noindex.iloc[i].signal == -1 ]
    buy_price_df = pd.DataFrame(buy_price_list,columns=['Date','buyprice']).set_index('Date')
    sell_price_df = pd.DataFrame(sell_price_list,columns=['Date','sellprice']).set_index('Date')
    stock_data['buyprice'] = buy_price_df['buyprice']
    stock_data['sellprice'] = sell_price_df['sellprice']

    # print("Signals Table:")
    # print(stock_data[stock_data['signal']!=0][['signal','buyprice','sellprice']])

    # print(f"Buy when Fast SMA({fast_sma}) CROSSES ABOVE the Slow SMA({slow_sma}))")
    # print(f"Sell when Fast SMA({fast_sma}) CROSSES BLOW the Slow SMA({slow_sma}))")

    signals_data = stock_data[stock_data['signal']!=0][['signal','buyprice','sellprice']]

    ret_signal = 'Sell'
    if len(signals_data)> 0:
        ret_date = signals_data.index[-1]
        ret_date = ret_date.strftime("%Y-%m-%d")
        if signals_data.iloc[-1].signal == 1:
            ret_signal = 'Buy'
    else:
        ret_signal = None
        ret_date = None
        
        
    return ret_signal, ret_date, (fast_sma,slow_sma)

    # plot_data = stock_data.iloc[-1000:].copy()
    # ax = plt.plot(plot_data['Close'], linewidth = 2)
    # plt.plot(plot_data[f'{fast_sma}_day_SMA'],linewidth = 2,c='red')
    # plt.plot(plot_data[f'{slow_sma}_day_SMA'],linewidth = 2,c='green')

    # # plt.plot(plot_data['signal'], color = 'green', linewidth = 2, label = 'ST UPTREND')
    # # plt.plot(stock_data['st_dt'], color = 'r', linewidth = 2, label = 'ST DOWNTREND')
    # plt.plot(plot_data.index, plot_data['buyprice'], marker = '^', color = 'green', markersize = 12, linewidth = 0, label = 'BUY SIGNAL')

    # plt.plot(plot_data.index, plot_data['sellprice'], marker = 'v', color = 'r', markersize = 12, linewidth = 0, label = 'SELL SIGNAL')
    # plt.title(stock+' SMA CROSSINGS TRADING SIGNALS')
    # plt.legend(['Close', f'{fast_sma}_day_SMA', f'{slow_sma}_day_SMA','Buy Signal','Sell Signal'],loc='upper left')
    # plt.show()
    
if __name__ == "__main__":
    stock = 'TSLA'
    data = yf.download(stock, start='2022-01-01', interval='1d',progress=False)
    ret_signal, ret_date, fast_slow = sma_xing(data,40,200)
    
    print(ret_signal,ret_date,fast_slow)

