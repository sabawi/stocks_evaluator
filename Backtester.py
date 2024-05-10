"""
    By Al Sabawi
    2023-03-11 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
import holidays
from termcolor import colored as cl


class SignalsBacktester:
    """This class provides functions to backtest a set of signals in a daily datetime indexed pandas 
    DatafFrame against a daily stock prices DataFrame and returns the resulted returns in another DataFrame 
    """
    def __init__(self,df_in,signals,start_date=None,end_date=None,amount=100000):
        """Initialization requires the daily prices which includes the datetime type daily index and the 'Close' column as floats

        Args:
            df_in (_type_): pandas DataFrame of the End-Of-Day (EOD) stock prices
            signals (_type_): pandas Series of 1's and 0's where 1=Buy, 0=Sell or Don't buy
            start_date (_type_, optional): String in the date format "%Y-%m-%d". Defaults to None.
            end_date (_type_, optional):  String in the date format "%Y-%m-%d". Defaults to None.
            amount (int, optional): int. The initial investment. Defaults to 100000.
        """
        self.df_in = df_in
        self.signals = pd.DataFrame(signals,columns=['Predicted'],index=self.df_in.index)
        self.print_details = True
        self.amount = amount
        self.tran_history = None
        
        # Fix the start and end dates to be Business days
        if start_date:
            self.start_date = self.next_business_day(start_date)
        else:
            self.start_date = self.df_in.index[0].strftime("%Y-%m-%d")
            
        if end_date:
            self.end_date = self.next_business_day(end_date)   
        else:
            self.end_date = self.df_in.index[-1].strftime("%Y-%m-%d")
            
        # print(f"Start Business Date = {self.start_date}")
        # print(signals)
            
    def set_print_details(self,print_details):
        self.print_details = print_details
        
    def set_tran_history(self, tran_history):
        self.tran_history = tran_history
    
    def get_tran_history(self):
        return self.tran_history
    
    def DatesRange(self,df):
        start = self.start_date
        end = self.end_date
        
        if start :
            start = pytz.utc.localize( datetime.fromisoformat(  start)).strftime("%Y-%m-%d")
        else:
            start = df.index[0]

        if end :
            end = pytz.utc.localize( datetime.fromisoformat( end)).strftime("%Y-%m-%d")
        else:
            end = pytz.utc.localize(datetime.today()).strftime("%Y-%m-%d")

        return df.loc[(df.index >= start) & (df.index <= end)].copy()

    def backtest(self):

        df_bt = self.df_in.join(self.signals, how='outer').fillna(method='ffill')
        df_bt = self.DatesRange(df_bt)
        
        df_transactions = pd.DataFrame(np.zeros((len(df_bt), 8)),
                                    columns=['Buy_Count','Buy_Amount','Sell_Count','Sell_Amount',
                                                'Shares_Count','Running_Balance','Account_Value','ROI_pcnt'],index=df_bt.index)
        # df_transactions.fillna(method='ffill')
        # print(df_transactions)
        
        # df_transactions.iloc[0].Running_Balance = self.amount
        for i in range(0,len(df_bt)):
            
            if i == 0:
                df_transactions.iloc[i].Running_Balance = self.amount
                df_transactions.iloc[i].Shares_Count = 0
            else:
                df_transactions.iloc[i].Running_Balance = df_transactions.iloc[i-1].Running_Balance
                df_transactions.iloc[i].Shares_Count = df_transactions.iloc[i-1].Shares_Count
            
            # print(f"df_bt.iloc[i-1] , date {df_bt.iloc[i-1]}, predicted {df_bt.iloc[i-1].Predicted}")
            
            # print(i, df_transactions.iloc[i].Running_Balance)
            if df_bt.iloc[i].Predicted == 1.0:
                shares_to_buy = int(df_transactions.iloc[i].Running_Balance / df_bt.iloc[i].Open)
                if(shares_to_buy > 0):
                    cost_of_shares = round(shares_to_buy * df_bt.iloc[i].Open,2)
                    df_transactions.iloc[i].Buy_Count = int(shares_to_buy)
                    df_transactions.iloc[i].Buy_Amount = cost_of_shares
                    df_transactions.iloc[i].Running_Balance = df_transactions.iloc[i].Running_Balance - cost_of_shares
                    # print(df_transactions.iloc[i+1].Running_Balance)
                    df_transactions.iloc[i].Shares_Count = int(shares_to_buy + df_transactions.iloc[i].Shares_Count)
            if df_bt.iloc[i].Predicted == 0.0:
                if df_transactions.iloc[i].Shares_Count > 0:
                    proceeds_of_sale = round(df_transactions.iloc[i].Shares_Count * df_bt.iloc[i].Open,2)
                    df_transactions.iloc[i].Sell_Amount = proceeds_of_sale
                    df_transactions.iloc[i].Running_Balance = df_transactions.iloc[i].Running_Balance + proceeds_of_sale
                    # print(df_transactions.iloc[i+1].Running_Balance)
                    df_transactions.iloc[i].Sell_Count = df_transactions.iloc[i].Shares_Count
                    df_transactions.iloc[i].Shares_Count = 0
            
            df_transactions.iloc[i].Account_Value = round(df_transactions.iloc[i].Running_Balance + \
                                                    (df_transactions.iloc[i].Shares_Count *  df_bt.iloc[i].Close),2)
            df_transactions.iloc[i].ROI_pcnt = round( ((df_transactions.iloc[i].Account_Value/self.amount) -1)*100,2)
            
            # print(f" {df_transactions.index[i]}")
        self.set_tran_history(df_transactions)
        
        # df_bt = df_bt.join(df_transactions, how='outer').fillna(method='ffill')
        df_bt = df_bt.join(df_transactions, how='outer')
        
        # print("df_bt",df_bt)
        return df_bt

    def next_business_day(self,date_str, country="US"):
        date = pd.to_datetime(date_str)
        holiday_dates = holidays.CountryHoliday(country, years=date.year).keys()
        while date.weekday() >= 5 or date.strftime("%Y-%m-%d") in holiday_dates:
            date = date + pd.tseries.offsets.BDay()
            
        if date not in self.df_in.index:
            for d in self.df_in.index:
                if d > date:
                    date = d
                    break
                
        return date.strftime("%Y-%m-%d")
    
    def buy_and_hold_strategy(self):
        # df_prices ,start_date=None, end_date=None, amount=100000
        # Buy and hold
             
        first_day_open = round(self.df_in.loc[self.start_date].Open,2)
        number_of_shares = int(self.amount/first_day_open)
        cost_of_shares = round(first_day_open * number_of_shares,2)
        cash_balance = round(self.amount - cost_of_shares,2)
        value_of_shares_last_day = round(number_of_shares * self.df_in.iloc[-1].Close,2)
        account_value = round(cash_balance+value_of_shares_last_day,2)
        buy_and_hold_ROI = round( ( float(account_value/self.amount) - 1.0)*100 ,2)
        
        if self.print_details:
            print("\n")
            print(f"First Day Share Price = ${round(first_day_open,2)}")
            print(f"Buy {number_of_shares} on {self.start_date}")
            print(f"Cost of Shares = ${cost_of_shares}")
            print(f"Remaining Cash Balance = {cash_balance}")
            print(f"Value of share on {self.df_in.index[-1]} = {value_of_shares_last_day}")
            print(f"Last Day Share Price = ${self.df_in.iloc[-1].Close}")
            print(f"Last Account Value = {account_value}")
            print(f"Buy and Hold ROI = {buy_and_hold_ROI}%")
        
        return buy_and_hold_ROI, account_value

    def plot_account(self,stock=None):
        buyhold_account = pd.DataFrame(index=self.df_in.index,columns=["BuyHoldValue"])
        buyhold_account = self.DatesRange(buyhold_account)
        
        close_changes = self.df_in.Close.pct_change().copy()
        close_changes = self.DatesRange(close_changes)
        
        buyhold_account.iloc[0].BuyHoldValue = self.amount
        # print(buyhold_account)
        
        for i in range(1,len(buyhold_account)):
            buyhold_account.iloc[i].BuyHoldValue = buyhold_account.iloc[i-1].BuyHoldValue + \
            (close_changes.iloc[i] * buyhold_account.iloc[i-1].BuyHoldValue)
        
        # print(buyhold_account)
        ax = self.tran_history[['Account_Value']][1:].plot(figsize=(15,8),grid=True)
        buyhold_account.plot(ax=ax)
        
        plt.grid(axis='both')
        plt.legend(title='Investment Accounts:')
        if stock:
            plt.title(f"{stock} AI vs Buy&Hold",fontsize = 14, fontweight ='bold')
        else:
            plt.title(f"AI vs Buy&Hold",fontsize = 14, fontweight ='bold')
            
        plt.show()
        
    def run(self):
        """Generate the backtesting calculations and return a pandas DataFrame with the results 
        """
        # invest_mount = 100000
        # start_date = '2022-09-01'

        print(f"Start Investment Date : {self.start_date}")
        print(f"Investment Amount : ${self.amount}")

        # df_tran = backtest(df_in,df_pred1.Predicted,start_date='2022-01-01',end_date='2023-01-01')
        df_tran = self.backtest()
        returns = df_tran.ROI_pcnt[-1]
        buy_and_hold_ROI,ending_account_value = self.buy_and_hold_strategy()
        if returns > buy_and_hold_ROI:
            on_color = 'on_green'
        else:
            on_color = 'on_red'
            
        print(cl(f"\nAI PREDICTION ENGINE Strategy:\n\tReturn on Investment : {returns}%\n\tEnding Account Value : ${df_tran['Account_Value'].iloc[-1]}",attrs=['bold'],on_color=on_color))

        # print(df_tran[['Open','Close','Predicted','Buy_Count','Buy_Amount',
        #          'Sell_Count','Sell_Amount','Shares_Count','Running_Balance','Account_Value','ROI_pcnt']].tail(30))

        print(f"\nBUY & HOLD Strategy:\n\tReturn on Investment : {buy_and_hold_ROI}%\n\tEnding Account Value : ${ending_account_value}")

    def results(self):
        tran_history = self.get_tran_history()
        # print(f"Number of Buys : {tran_history['Buy_Count'].sum()}")
        # print(f"tran_history: \n Buy Count:{tran_history.Buy_Count>0}\n{tran_history}")
        max_account_value = tran_history[(tran_history['Account_Value'] == tran_history['Account_Value'].max())]
        if tran_history['Buy_Count'].sum() > 0:
            first_buy = tran_history[(tran_history['Buy_Count']>0)].index[0].strftime("%Y-%m-%d")
        else:
            first_buy = 'None'
        buys = len(tran_history[tran_history['Buy_Count']>0])
        sells = len(tran_history[tran_history['Sell_Count']>0])
        days_bought_or_sold = tran_history[(tran_history['Buy_Count'] > 0) | (tran_history['Sell_Count'] > 0)]

        # If no transactions, exit now else show transaction table
        if buys == 0 :
            return
        
        print(cl("************************************************************",attrs=['bold']))
        print(cl("**                        RESULTS                         **",attrs=['bold'],on_color='on_blue'))       
        print(cl("************************************************************",attrs=['bold']))
        print("\nFirst Buy On:",first_buy)
        print(f"Number of Buy Transactions : {buys}")
        print(f"Number of Sell Transactions : {sells}")
        print(f"Number of Days Transactions Happened : {len(days_bought_or_sold)} out of {len(tran_history)} Trading Days")

        trans_from_first_buy = tran_history.loc[first_buy:]
        lowest_account_value = trans_from_first_buy[(trans_from_first_buy['Account_Value'] == trans_from_first_buy['Account_Value'].min())]
        max_drawdown = tran_history[(tran_history['ROI_pcnt'] == tran_history['ROI_pcnt'].min())]
        highest_ROI = tran_history[(tran_history['ROI_pcnt'] == tran_history['ROI_pcnt'].max())]

        print(f"\nPeak Account Value was on {max_account_value.index[0].strftime('%Y-%m-%d')} : ${max_account_value.iloc[0].Account_Value}")
        print(f"Lowest Account Value was on {lowest_account_value.index[0].strftime('%Y-%m-%d')} : ${lowest_account_value.iloc[0].Account_Value}")

        print(f"\nHighest ROI (Max % Return) was on {highest_ROI.index[0].strftime('%Y-%m-%d')} : {highest_ROI.iloc[0].ROI_pcnt}%")
        print(f"Lowest ROI (Max % Drawdown) was on {max_drawdown.index[0].strftime('%Y-%m-%d')} : {max_drawdown.iloc[0].ROI_pcnt}%")

        print(cl("************************************************************",attrs=['bold']))
        print(cl("**                   TRANSACTION TABLE                    **",attrs=['bold'],on_color='on_blue')) 
        print(cl("************************************************************",attrs=['bold']))
        print(tran_history)

if __name__ == "__main__":
    pass