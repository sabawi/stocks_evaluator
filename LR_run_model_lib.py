#!/usr/bin/env python
# coding: utf-8

# In[3]:


"""_Al Sabawi
    2023-03-13
    summary_
        Backtest a saved Logistic Regression Model and Predicts the next 5 days
    Returns:
        _Printed output
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pickle
from Backtester import SignalsBacktester as bt
import datetime
from termcolor import colored as cl


def date_n_years_ago(N):
    today = datetime.date.today()
    years_ago = datetime.timedelta(days=N*365)
    return today - years_ago

def create_lagged_features(df, lag):
    features = df.copy()
    for i in range(1, lag+1):
        features[f'ret_lag{i}'] = features['Close'].shift(i)
    features.dropna(inplace=True)
    return features

def get_next_day (df_lagged,model):
    # import datetime
    # from pandas.tseries.offsets import BDay  
    df_lagged2 = df_lagged.copy()

    # last_date = df_lagged2.index[-1]
    # next_day_date = (last_date + BDay(1)).strftime("%Y-%m-%d")
    # weekday = (last_date + BDay(1)).strftime("%A")

    next_day_predication_ml1 = model.predict(df_lagged2.iloc[[-1]])
    
    # if next_day_predication_ml1 > 0:
    #     recomm = f"Buy at the Open (Prediction = {next_day_predication_ml1[0]})"
    # else:
    #     recomm = f"Don't Buy at the Open (Prediction = {next_day_predication_ml1[0]})"
        
    # print(cl("*********************************************************************************", attrs=['bold']))
    # print(cl(f"On {weekday} {next_day_date}, AI's Recommendation is: {recomm}", attrs=['bold']))
    # print(cl("*********************************************************************************", attrs=['bold']))

    return next_day_predication_ml1[0]

# ###########################################################################################
# ###########################################################################################

# Load up date for Stock
# stock = input("Enter the Stock symbol to analyze: ")
# stock = stock.upper()
# years_ago = input("Backtesting Lookback Years (1 or more) :")

def get_recommendation(stock,lookback):
    years_ago = int(lookback)
    start_date = date_n_years_ago(years_ago).strftime("%Y-%m-%d")
    end_date = None
    df_in = yf.download(stock,period='max',progress=False)
    # df_in = yf.Ticker(stock).history(start=start_date,interval='1d',period='max')
    # print(f"Models for: {stock}")
    # Generate the pct_change for the closing prices
    df_in_delta = df_in.copy()
    df_in_delta['Close'] = df_in_delta['Close'].pct_change() 

    lag_depth = 30  # MUST BE the same as the training dataset
    # Create the Lagged Dataset
    df_lagged = create_lagged_features(df_in_delta,lag_depth)

    # Remove the non-features from the Dataset
    if 'Adj Close' in df_lagged:
        df_lagged = df_lagged.drop(['Open','High','Low','Close','Adj Close','Volume'],axis=1)
    else:
        df_lagged = df_lagged.drop(['Open','High','Low','Close','Volume'],axis=1)

    recomm_list = []

    for j in [1,2]:
        # print( cl("***************************************************************",attrs=['bold']))
        # print(cl(f"**                        {stock} Model {j}                         ",attrs=['bold'],color='white',on_color='on_blue'))
        # print( cl("***************************************************************",attrs=['bold']))
        # Load the LR Model from disk
        with open(f'./models/{stock}_last_model{j}.pkl', 'rb') as f:
            LR_model = pickle.load(f)
            
        # Predict all the days in time series with the model
        # y_pred = LR_model.predict(df_lagged)
        # df_pred = pd.DataFrame(y_pred,columns=['Predicted'],index=df_lagged.index)

        # Run backtesting on the model to verify the results
        # backtest = bt(df_in=df_in, signals=df_pred, start_date=start_date, end_date=None, amount=100000)
        # backtest.run()
        # tran_history = backtest.get_tran_history()

        # backtest.results()
        # backtest.plot_account(f"{stock} Model {j}")

        recomm_list.append(get_next_day (df_lagged = df_lagged, model = LR_model))
        
        
    #  Best Model
    # print( cl("***************************************************************",attrs=['bold']))
    # print(cl(f"**                        {stock} BEST MODEL                        ",attrs=['bold'],color='white',on_color='on_blue'))
    # print( cl("***************************************************************",attrs=['bold']))
    # Load the LR Model from disk
    with open(f'./models/{stock}_best_model.pkl', 'rb') as f:
        LR_model = pickle.load(f)
        
    # Predict all the days in time series with the model
    # y_pred = LR_model.predict(df_lagged)
    # df_pred = pd.DataFrame(y_pred,columns=['Predicted'],index=df_lagged.index)

    # # Run backtesting on the model to verify the results
    # backtest = bt(df_in=df_in, signals=df_pred, start_date=start_date, end_date=None, amount=100000)
    # backtest.run()
    # tran_history = backtest.get_tran_history()

    # backtest.results()
    # backtest.plot_account(f"{stock} Best Model")

    recomm_list.append(get_next_day (df_lagged=df_lagged, model=LR_model))

    # ### Detect Multiple Buy Transactions Before Selling

    # In[4]:


    # Find a buy without a sell
    # buy = False
    # extra_buys = 0
    # for i in range(len(tran_history)):
    #     if tran_history.iloc[i].Buy_Count > 0:
    #         if buy :
    #             extra_buys +=1
    #             print(f"{extra_buys} Extra Buy(s) Detected without a prior Sell in a row at {i}")
    #             print("Date : ",tran_history.index[i].strftime("%Y-%m-%d"))
    #         else:
    #             buy = True
    #     if tran_history.iloc[i].Sell_Count > 0:
    #         if buy:
    #             buy = False
                
    # print("Detecting Multiple Buy Transactions Before Selling:")
    # if extra_buys == 0:
    #     print("No extra buys detected")
        
        
    # print(recomm_list)
    labels = ['Sell','Buy']
    rlist = [labels[int(x)] for x in recomm_list]
    # rlist = (rlist[0],rlist[1],rlist[2])
    return rlist


if __name__ == "__main__":
    print(get_recommendation(stock='ADI',lookback=2))
