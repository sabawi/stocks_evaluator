#!/usr/bin/env python
# coding: utf-8

# ## Linear Logistics Models
# ### for 
# ## Predicting Stock Market Movements
# The moves UP or Down are classified only

# In[337]:


"""
    By Al Sabawi
    # 2023-03-11 
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import yfinance as yf
import matplotlib.pyplot as plt
import pytz


def create_model(stock):
    # Load stock price data
    df_in = yf.download(stock,period='max',progress=False)


    # ### Latest Stock Dataset from Yahoo Finance  

    # In[338]:


    # Create an isolated dataframe so we can manipulate without changing the original data
    df = df_in.copy()

    # Drop all the columns not relevant for the daily predictions
    df = df.drop(columns=['Open','High','Low','Adj Close','Volume'])

    # Create a 'Returns' column for the % changes in price and add it to Dataframe
    df['ret'] = df['Close'].pct_change()   

    # The 'Target' is what we will predict in LogisticRegression 
    # The Target is 1 or 0, 1 meaning the stock went up TODAY from Yesterday's price
    # However, since we need to predict it a day ahead so we can buy it, we need to shift() back in time!
    # so we get the signal to buy before the day the price goes up
    ## The following line says: If tomorrow's return 'df['ret'].shift(-1)' is above 0, record a buy signal (1) 
    # today so we buy it at the open tomorrow, else record 'no buy' signal (0)
    df['Target'] = np.where(df['ret'].shift(-1) > 0, 1,0)
    df.at[df.index[-1],'Target'] = np.nan
    df = df.dropna()


    # In[339]:


    # print(f"Last Day : {df.index[-1]}")

    # df[['Close','ret','Target']].tail(10)


    # ### Creating LAGGED Dataset

    # In[340]:


    # A lagged dataset in Timeseries is based on the assumption that the predicted value 'Target' 
    # depends on the prices of 1 or more days before.  In this case I am taking into account 'lag_depth' days before
    # We will add 5 new columns recording the change in price for the past 5 days in each row

    # Create lagged features for the past lag_depth days
    lag_depth = 30
    def create_lagged_features(df, lag):
        features = df.copy()
        for i in range(1, lag+1):
            features[f'ret_lag{i}'] = features['ret'].shift(i)
        features.dropna(inplace=True)
        features.drop(columns=['Close'],inplace=True)
        features.drop(columns=['ret'],inplace=True)
        return features

    df_lagged = create_lagged_features(df, lag_depth)
    df_lagged.tail(6)


    # ### Training Set and Batches
    # ##### We'll need to divide the historical data in smaller batches but we need to make sure each batch is balanced as much as possible

    # In[341]:


    df_lagged.dropna(inplace=True)
    randomized = False # If true, we will get batches with randomized rows BUT Balanced which means almost equal number of 1 and 0 Target

    # ##############################################################
    # About Batches:    For a LogisticRegression Model, we need to 
    #                   balance the training data with rows that have 
    #                   equal 'Target' of 1 (buy) and 0 (no buy). 
    #                   Otherwise the model will become bias for the outcome
    #                   that we feed it more of.  So for that we made each 
    #                   row 'self-contained' with all the previous 
    #                   data (last return plu 5 previous returns) so that 
    #                   we can shuffle the rows and feed them into 
    #                   the model as batches of rows. Each bach is an equal 
    #                   mix of outcome 1 and 0.  This was the concentration 
    #                   of 1's and 0's dont in a series (long up trends or 
    #                   long down trends) don't bias the next outcome
    # ###############################################################
    batches_count = 1  # We can start from 32 to go up then see the accuracy the effect of accuracy
    train_batches = []

    if randomized: # Make it randomized ONLY if the order of data does not matter. In market data, IT DOES MATTER
        # Split data into train and test sets using a stratified 80-20 split
        train_df, test_df = train_test_split(df_lagged, test_size=0.2, random_state=42, stratify=df_lagged['Target'])
        # Split train data into batches with balanced target values
        batch_size = len(train_df) // batches_count

        for i in range(0, len(train_df), batch_size):
            batch = train_df.iloc[i:i+batch_size]
            num_positives = len(batch[batch['Target'] == 1])
            if num_positives == batch_size // 2:
                train_batches.append(batch)
            elif num_positives > batch_size // 2:
                excess_positives = num_positives - batch_size // 2
                batch = batch.drop(batch[batch['Target'] == 1].sample(excess_positives).index)
                train_batches.append(batch)
            else:
                missing_positives = batch_size // 2 - num_positives
                num_negatives = len(batch[batch['Target'] == 0])
                if missing_positives > num_negatives:
                    batch = batch.drop(batch[batch['Target'] == 0].index)
                    missing_positives -= num_negatives
                    excess_positives = missing_positives - len(batch[batch['Target'] == 1])
                    batch = pd.concat([batch, batch[batch['Target'] == 1].sample(excess_positives, replace=True)])
                else:
                    batch = batch.drop(batch[batch['Target'] == 0].sample(missing_positives, replace=False).index)
                train_batches.append(batch)
    else:
        train_df, test_df = train_test_split(df_lagged,test_size=0.1, shuffle=False)
        batch_size = len(train_df) // batches_count
        for i in range(0, len(train_df), batch_size):
            train_batch = train_df.iloc[i:i+batch_size]
            train_batches.append(train_batch)


    # print(f"Number of Batches = {batches_count}")
    # print(f"Rows in train first batch = {len(train_batches[0])}")
    # train_batches[0].tail(len(train_batches[0]))
    train_batches[0]


    # In[342]:


    # train_df dataframe is the unbatched dataset
    train_df.tail(5)


    # ### model_1 : Create the 1st Model

    # Possible Parameters:
    # solver: This parameter determines the algorithm used to solve the optimization problem. For large datasets, it is often beneficial to use a solver that is specifically designed for large-scale problems, such as lbfgs, sag, or saga.
    # 
    # max_iter: This parameter sets the maximum number of iterations for the solver. For large datasets, it may be necessary to increase this parameter to ensure that the solver converges to a solution.
    # 
    # penalty: This parameter determines the type of regularization used in the model. Regularization can help prevent overfitting and improve model generalization. The l1 and l2 penalties are commonly used, but for large datasets, the elasticnet penalty may be more appropriate as it combines both l1 and l2 regularization.
    # 
    # C: This parameter sets the inverse regularization strength. A smaller value of C will increase the regularization strength, which can help prevent overfitting. For large datasets, a smaller value of C may be more appropriate.
    # 
    # class_weight: This parameter allows for adjusting the weights of different classes in the dataset. This can be useful when working with imbalanced datasets, where one class is much more prevalent than the others.

    # In[343]:


    # Create logistic regression model
    # model_1 = LogisticRegression(class_weight='balanced',solver='saga',
    #                              max_iter=200,penalty='l2',C=0.5)
    model_1 = LogisticRegression()
    param_grid = {
        'C' : [100, 10, 1.0, 0.1, 0.01],
        # 'C': [0.1, 1, 10, 100],
        # 'solver': ['lbfgs', 'liblinear', 'sag', 'saga'],
        'solver' : ['newton-cg', 'lbfgs', 'liblinear','newton-cholesky'],
        'penalty' : ['l2'],
        'max_iter': [100, 500, 1000, 5000],
        'class_weight': ['balanced']
    }

    # Create our GridSearchCV object and fit it to our training data
    grid_search = GridSearchCV(estimator=model_1, param_grid=param_grid, cv=5)


    # ### Testing the model (model_1)

    # In[344]:


    # Train model on the first batch of the training data
    X_train = train_batches[0].drop(columns=['Target'])
    y_train = train_batches[0]['Target']
    # print("X_train: \n",X_train.tail(5))
    # print("y_train: \n",y_train.tail(5))
    # ******************************************************
    grid_search.fit(X_train, y_train)
    # grid_search.partial_fit(X_train, y_train, classes=[0, 1])
    # Evaluate the model on the test set and print the test accuracy
    X_test = test_df.drop(columns=['Target'])
    y_test = test_df['Target']
    # y_pred = model_1.predict(X_test)

    # Retrieve the best hyperparameters and the best model from the GridSearchCV object
    best_params = grid_search.best_params_
    # print(best_params)
    best_model = grid_search.best_estimator_

    # Use the best model to make predictions on our test set and evaluate its accuracy
    y_pred = best_model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)
    # print(f'Test accuracy: {test_accuracy:.3f}')


    # In[345]:


    # Show predictions
    y_pred


    # ### Train the remaining batches on model_1

    # In[346]:


    # print("BatchSize",batch_size)
    for j in range(0,batches_count):
        # print("Batch#",j,train_batches[j])
        X_train = train_batches[j].drop(columns=['Target'])
        # print(X_train.columns)
        y_train = train_batches[j]['Target']
        # model_1.fit(X_train, y_train)
        best_model.fit(X_train, y_train)

    # X_train = train_df.drop(columns=['Target'])
    # y_train = train_df['Target']

    # #***************************
    # model.fit(X_train, y_train)
    # X_train


    # ### Predictions from model_1

    # In[347]:


    # Make predictions for next 5 days
    first_value = df_in['Close'][0]
    df_pred = df_in.copy()
    # print(first_value)


    # ### Use the whole dataset and removing the Non-Feature columns
    # Non-feature columns are columns not used for training

    # In[348]:


    # Generate the Retuen columns and the Target column to compare with later
    df_pred['ret'] = df_pred['Close'].pct_change() # Daily return
    df_pred['Target'] = np.where(df_pred['ret'].shift(-1) > 0.0, 1, 0) # Target column is 1 IF the next day close price is higher
    df_pred.at[df.index[-1],'Target'] = np.nan
    df_pred = df_pred.dropna()

    df_pred = create_lagged_features(df_pred, lag_depth) # Create the Lagged columns for the past 5 days

    # df_pred.drop(columns=['Target'],inplace=True)
    df_pred = df_pred.drop(columns=['Open','High','Low','Adj Close','Volume']) # Remove non-feature columns
    if 'Predicted' in df_pred.columns:
        df_pred = df_pred.drop('Predicted') # Remove the predicted column in case its leftover from previous runs
    df_pred.tail(5)


    # In[349]:


    # Create a separate dataframe WITHOUT target column for prediction only
    data_no_target = df_pred.copy()
    if('Target' in data_no_target.columns):
        data_no_target = data_no_target.drop(columns=['Target'])

    # Check to see if we have the right no. of columns for the prediction call
    # print('column count =',len(data_no_target.columns),':',data_no_target.columns)
    # Check that we have the Target data still available
    df_pred['Target'].tail(5)


    # In[350]:


    # predictions_1 = model_1.predict(data_no_target)
    predictions_1 = best_model.predict(data_no_target)


    # In[351]:


    # Make predictions from model_1 into a DataFrame along with the actual Target column from before to compare
    df_pred1 = pd.DataFrame(index=df_pred.index)
    df_pred1['Predicted']=  predictions_1
    df_pred1['Target'] = df_pred['Target'].copy()
    df_pred1[['Predicted','Target']].tail(50)


    # ### Check Prediction Results for model_1

    # In[352]:


    eq=neq=pup=tup=pdown=tdown=0
    for i in range(len(df_pred1['Predicted'])):
        if df_pred1['Predicted'].iloc[i] == df_pred1['Target'].iloc[i]:
            eq+=1
            if df_pred1['Predicted'].iloc[i] == 1:
                pup+=1
                tup+=1
            else:
                pdown+=1
                tdown+=1
        else:
            neq+=1
            if df_pred1['Predicted'].iloc[i] == 1:
                pup+=1
                tdown+=1
            if df_pred1['Target'].iloc[i] == 1:
                pdown+=1
                tup+=1
        
    model1_correctness = round(100*eq/(eq+neq),2)
    # print("----Results from Predictions using model_1----")  
    # print(f"Equal Values = {eq} ({model1_correctness}%) \n\
    # Not Equal = {neq} ({round(100*neq/(eq+neq),2)}%),  \n\
    # Total = {eq+neq} rows")
    # print(f"Predicted UPs : {round(100*pup/(eq+neq),2)}% vs Actual UPs : {round(100*tup/(eq+neq),2)}%  ")
    # print(f"Predicted Downs : {round(100*pdown/(eq+neq),2)}% vs Actual Downs : {round(100*tdown/(eq+neq),2)}%  ")



    # ### model_2: Creating the second model
    # This model will be trained without batches or manual re-balancing of outcomes 

    # In[353]:


    # Creating model_2 and training it on the whole dataset on one go. No batching or rebalancing 
    model_2 = LogisticRegression(class_weight='balanced')
    # print('Check columns : ',data_no_target.columns)
    model_2.fit(data_no_target, df_pred['Target'])
        
    df_pred2 = pd.DataFrame(index=df_pred.index)
    df_pred2['Predicted']=  model_2.predict(data_no_target)
    df_pred2['Target'] = df_pred['Target'].copy()
    df_pred2[['Predicted','Target']].tail(50)


    # In[354]:


    # Reassembling the original dataset with the Predicted and Target columns added
    df_in2 = df_in.copy()
    df_in2['Predicted Buy'] = df_pred2['Predicted']
    df_in2['Correct Buy'] = df_pred2['Target']
    df_in2.dropna()
    df_in2.tail(20)
    # df_pred['Target'].tail(50)


    # In[355]:


    df_in2.tail(5)


    # ### Check Prediction Results for model_2

    # In[356]:


    eq=neq=pup=tup=pdown=tdown=0
    for i in range(len(df_pred2['Predicted'])):
        if df_pred2['Predicted'].iloc[i] == df_pred2['Target'].iloc[i]:
            eq+=1
            if df_pred2['Predicted'].iloc[i] == 1:
                pup+=1
                tup+=1
            else:
                pdown+=1
                tdown+=1
        else:
            neq+=1
            if df_pred2['Predicted'].iloc[i] == 1:
                pup+=1
                tdown+=1
            if df_pred2['Target'].iloc[i] == 1:
                pdown+=1
                tup+=1
        
    model2_correctness = round(100*eq/(eq+neq),2)
    # print("----Results from Predictions using model_2----")  
    # print(f"Equal Values = {eq} ({model2_correctness}%) \n\
    # Not Equal = {neq} ({round(100*neq/(eq+neq),2)}%),  \n\
    # Total = {eq+neq} rows")
    # print(f"Predicted UPs : {round(100*pup/(eq+neq),2)}% vs Actual UPs : {round(100*tup/(eq+neq),2)}%  ")
    # print(f"Predicted Downs : {round(100*pdown/(eq+neq),2)}% vs Actual Downs : {round(100*tdown/(eq+neq),2)}%  ")


    # ### Predicting the Stock Market for the next 5 Days 
    # We'll use model_2 and follow the same procedures of no batches or re-balancing 

    # In[357]:


    df_pred = df_in.copy()
    # df_pred['Close'].pct_change()
    df_pred['ret'] = df_pred['Close'].pct_change()
    df_pred['Target'] = np.where(df_pred['ret'].shift(-1) > 0.0, 1, 0)
    df_pred = create_lagged_features(df_pred, lag_depth)
    df_pred


    # In[358]:


    df_pred = df_pred.drop(columns=['Open','High','Low','Adj Close','Volume'])
    df_pred


    # ### 5-Days in the future stock predictions

    # In[359]:


    # We need al least 5 days from the past without the Target column
    last_five_days = df_pred.iloc[-5:].copy()
    last_five_days.drop('Target',inplace=True,axis=1)

    # We need to add the Predicted column for future predictions
    new_columns = last_five_days.columns[-lag_depth:].to_list()
    # new_columns.append('model_1_Prediction')
    new_columns.append('best_model_Prediction')
    new_columns.append('model_2_Prediction')

    # We need to prepare an empty dataframe to receive future data
    next_five_days = pd.DataFrame(columns=new_columns)

    # Now starting from the first of the last 5 days, predict tomorrow Up or Down market, then move forward one day
    for i in range(1, 6):
        # next_day_m1 = model_1.predict(last_five_days.iloc[[i-1]])
        next_day_m1 = best_model.predict(last_five_days.iloc[[i-1]])
        next_day_m2 = model_2.predict(last_five_days.iloc[[i-1]])
        # next_day = model.predict(last_five_days.iloc[i-1, 1:].values.reshape(1, -1))
        arr_m1 = np.append(last_five_days.iloc[i-1, :].values, next_day_m1[0])
        arr_m2 = np.append(arr_m1,next_day_m2[0])
        arr_df = pd.DataFrame([arr_m2], columns=new_columns)
        next_five_days= pd.concat([next_five_days,arr_df])
        
    # Create the next 5 working dates and make an index for the predicted 5 days
    import datetime
    from pandas.tseries.offsets import BDay
    # daysdates = [(datetime.datetime.today() + BDay(i)).strftime("%Y-%m-%d")  for i in range(1,6) ]
    # df_next5days = pd.DataFrame(next_five_days)
    # df_next5days.index = daysdates

    last_date = df_lagged.index[-1]
    daysdates = [(last_date + BDay(i)).strftime("%Y-%m-%d")  for i in range(1,6) ]
    df_next5days = pd.DataFrame(next_five_days)
    df_next5days.index = daysdates

    df_next5days['Week_Day'] = pd.Series([ (last_date + BDay(i)).strftime("%A") for i in range(1,6)],index=daysdates)
    buysell = ['NoBuy@Open','Buy@Open']
    buyselllist1, buyselllist2 = [],[]
    # for a in df_next5days['model_1_Prediction']:
    for a in df_next5days['best_model_Prediction']:
        buyselllist1.append(buysell[int(a)])

    for a in df_next5days['model_2_Prediction']:
        buyselllist2.append(buysell[int(a)])
        
    df_next5days['model_1_Action'] = pd.Series(buyselllist1,index=daysdates)
    df_next5days['model_2_Action'] = pd.Series(buyselllist2,index=daysdates)

    # print('\nPredictions for next 5 days:')
    # print(df_next5days[['Week_Day','best_model_Prediction','model_1_Action','model_2_Prediction','model_2_Action']])


    # In[360]:


    # Based on the above, Model_1 is predicting a BUY signal at End of day 2023-03-15, which means to BUY the Open on 2023-03-16
    # While Model_2 is predicting to BUY the morning Open the next trading day to 2023-03-17 which 2023-03-20


    # ### Save the Models

    # In[361]:


    import pickle
    # Save model_1

    # model1_fname = f'./models/LogisticRegressionModel1_{stock}_{datetime.datetime.today().strftime("%Y_%m_%dT%I%M%S%p")}.pkl'
    model1_fname = f'./models/LogisticRegressionBestModel_{stock}_{datetime.datetime.today().strftime("%Y_%m_%dT%I%M%S%p")}.pkl'
    model2_fname = f'./models/LogisticRegressionModel2_{stock}_{datetime.datetime.today().strftime("%Y_%m_%dT%I%M%S%p")}.pkl'
    last_mode1 = f'./models/{stock}_last_model1.pkl'
    last_mode2 = f'./models/{stock}_last_model2.pkl'

    # print("################# Saving the Models ###################")

    import os
    model_dir = './models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
   
    with open(model1_fname, 'wb') as f:
        pickle.dump(best_model, f)

    with open(last_mode1, 'wb') as f:
        pickle.dump(best_model, f)
        
    # print(f"Model {model1_fname} Saved!")
    # print(f"Model {last_mode1} Saved!")

    # Save Model_2
    with open(model2_fname, 'wb') as f:
        pickle.dump(model_2, f)

    with open(last_mode2, 'wb') as f:
        pickle.dump(model_2, f)
        
    # print(f"Model {model2_fname} Saved!")
    # print(f"Model {last_mode2} Saved!")
    # print("#######################################################")


    # ### Backtesting the AI Models

    # In[362]:


    from datetime import datetime

    def DatesRange(df, start=None, end=None):
        if start :
            start = pytz.utc.localize( datetime.fromisoformat(  start)).strftime("%Y-%m-%d")
        else:
            start = df.index[0]

        if end :
            end = pytz.utc.localize( datetime.fromisoformat( end)).strftime("%Y-%m-%d")
        else:
            end = pytz.utc.localize(datetime.today()).strftime("%Y-%m-%d")

        return df.loc[(df.index >= start) & (df.index <= end)].copy()

    def backtest(df_prices, buy_sell_signals, start_date=None, end_date=None,amount=100000):
        
        df_bt = df_prices.join(buy_sell_signals, how='outer').fillna(method='ffill')
        
        df_bt = DatesRange(df_bt, start=start_date, end=end_date)
        df_transactions = pd.DataFrame(np.zeros((len(df_bt), 8)),
                                    columns=['Buy_Count','Buy_Amount','Sell_Count','Sell_Amount',
                                                'Shares_Count','Running_Balance','Account_Value','ROI_pcnt'],index=df_bt.index)
        # df_transactions.fillna(method='ffill')
        # print(df_transactions)
        
        df_transactions.iloc[0].Running_Balance = amount
        for i in range(1,len(df_bt)):
            df_transactions.iloc[i].Running_Balance = df_transactions.iloc[i-1].Running_Balance
            df_transactions.iloc[i].Shares_Count = df_transactions.iloc[i-1].Shares_Count
            
            # print(i, df_transactions.iloc[i].Running_Balance)
            if df_bt.iloc[i-1].Predicted == 1.0:
                shares_to_buy = int(df_transactions.iloc[i].Running_Balance / df_bt.iloc[i].Open)
                if(shares_to_buy > 0):
                    cost_of_shares = round(shares_to_buy * df_bt.iloc[i].Open,2)
                    df_transactions.iloc[i].Buy_Count = int(shares_to_buy)
                    df_transactions.iloc[i].Buy_Amount = cost_of_shares
                    df_transactions.iloc[i].Running_Balance = df_transactions.iloc[i].Running_Balance - cost_of_shares
                    # print(df_transactions.iloc[i+1].Running_Balance)
                    df_transactions.iloc[i].Shares_Count = int(shares_to_buy + df_transactions.iloc[i].Shares_Count)
            if df_bt.iloc[i-1].Predicted == 0.0:
                if df_transactions.iloc[i].Shares_Count > 0:
                    proceeds_of_sale = round(df_transactions.iloc[i].Shares_Count * df_bt.iloc[i].Open,2)
                    df_transactions.iloc[i].Sell_Amount = proceeds_of_sale
                    df_transactions.iloc[i].Running_Balance = df_transactions.iloc[i].Running_Balance + proceeds_of_sale
                    # print(df_transactions.iloc[i+1].Running_Balance)
                    df_transactions.iloc[i].Sell_Count = df_transactions.iloc[i].Shares_Count
                    df_transactions.iloc[i].Shares_Count = 0
            
            df_transactions.iloc[i].Account_Value = round(df_transactions.iloc[i].Running_Balance + \
                                                    (df_transactions.iloc[i].Shares_Count *  df_bt.iloc[i].Close),2)
            df_transactions.iloc[i].ROI_pcnt = round( ((df_transactions.iloc[i].Account_Value/amount) -1)*100,2)
            
        # df_bt = df_bt.join(df_transactions, how='outer').fillna(method='ffill')
        df_bt = df_bt.join(df_transactions, how='outer')
        
        # print(df_buy)
        return df_bt

    def buy_and_hold_strategy(df_prices,start_date=None, end_date=None, amount=100000):
        # Buy and hold
        first_day_open = round(df_in.loc[start_date].Open,2)
        number_of_shares = int(amount/first_day_open)
        cost_of_shares = round(first_day_open * number_of_shares,2)
        cash_balance = round(amount - cost_of_shares,2)
        value_of_shares_last_day = round(number_of_shares * df_in.iloc[-1].Close,2)
        account_value = round(cash_balance+value_of_shares_last_day,2)
        buy_and_hold_ROI = round( ( float(account_value/amount) - 1.0)*100 ,2)
        
        print_details = False
        if print_details:
            print("\n")
            print(f"First Day Share Price = ${round(first_day_open,2)}")
            print(f"Buy {number_of_shares} on {start_date}")
            print(f"Cost of Shares = ${cost_of_shares}")
            print(f"Remaining Cash Balance = {cash_balance}")
            print(f"Value of share on {df_in.index[-1]} = {value_of_shares_last_day}")
            print(f"Last Day Share Price = ${df_in.iloc[-1].Close}")
            print(f"Last Account Value = {account_value}")
            print(f"Buy and Hold ROI = {buy_and_hold_ROI}%")
        
        return buy_and_hold_ROI, account_value


        
    invest_mount = 10000
    start_date = '2018-01-04'

    # print("\n")
    # print("########################################################")
    # print("#                    BACKTESTING                       #")
    # print("########################################################")

    # print(f"Start Investment Date : {start_date}")
    # print(f"Investement Amount : ${invest_mount}")


    # #########################################################################################
    # # ####################### MODEL model_1 #################################################
    # print("\n####################### MODEL model_1 #####################")
    # # df_tran = backtest(df_in,df_pred1.Predicted,start_date='2022-01-01',end_date='2023-01-01')
    df_tran = backtest(df_in,df_pred1.Predicted,start_date=start_date,amount=invest_mount)
    returns = df_tran.ROI_pcnt[-1]
    returns_model1 = returns


    # print(f"\nAI PREDICTION ENGINE Strategy:\n\tReturn on Investment : {returns}%\n\tEnding Account Value : ${df_tran['Account_Value'].iloc[-1]}")

    # # print(df_tran[['Open','Close','Predicted','Buy_Count','Buy_Amount',
    # #          'Sell_Count','Sell_Amount','Shares_Count','Running_Balance','Account_Value','ROI_pcnt']].tail(30))

    # buy_and_hold_ROI,ending_account_value = buy_and_hold_strategy(df_in,start_date=start_date, end_date=None, amount=invest_mount)
    # print(f"\nBUY & HOLD Strategy:\n\tReturn on Investment : {buy_and_hold_ROI}%\n\tEnding Account Value : ${ending_account_value}")

    # #########################################################################################
    # # ####################### MODEL model_2 #################################################
    # print("\n####################### MODEL model_2 #####################")
    # # df_tran = backtest(df_in,df_pred1.Predicted,start_date='2022-01-01',end_date='2023-01-01')
    df_tran = backtest(df_in,df_pred2.Predicted,start_date=start_date,amount=invest_mount)
    returns = df_tran.ROI_pcnt[-1]
    returns_model2 = returns

    # print(f"\nAI PREDICTION ENGINE Strategy:\n\tReturn on Investment : {returns}%\n\tEnding Account Value : ${df_tran['Account_Value'].iloc[-1]}")

    # # print(df_tran[['Open','Close','Predicted','Buy_Count','Buy_Amount',
    # #          'Sell_Count','Sell_Amount','Shares_Count','Running_Balance','Account_Value','ROI_pcnt']].tail(30))

    # buy_and_hold_ROI,ending_account_value = buy_and_hold_strategy(df_in,start_date=start_date, end_date=None, amount=invest_mount)
    # print(f"\nBUY & HOLD Strategy:\n\tReturn on Investment : {buy_and_hold_ROI}%\n\tEnding Account Value : ${ending_account_value}")


    # In[363]:


    # from Backtester import SignalsBacktester as bt

    # invest_mount = 100000
    # # start_date = '2018-03-15'

    # print("\n")
    # print("#########################################################")
    # print("#                           PLOTTING                    #")
    # print("#########################################################")

    # print("##### Model 1 #######")
    # backtest = bt(df_in=df_in, signals=df_pred1.Predicted, start_date=start_date, end_date=None, amount=invest_mount)
    # backtest.run()
    # backtest.plot_account(f"{stock} Model 1")

    # print("##### Model 2 #######")
    # backtest = bt(df_in=df_in, signals=df_pred2.Predicted, start_date=start_date, end_date=None, amount=invest_mount)
    # backtest.run()
    # backtest.plot_account(f"{stock} Model 2")


    # In[364]:


    best_model_filename = f'./models/{stock}_best_model.pkl'
        
    if returns_model1 > returns_model2:
        # print(f"Model 1 outperforms Model 2 by {(returns_model1 - returns_model2)}%")
        model_num = 1
        winner_model = best_model
    else:
        # print(f"Model 2 outperforms Model 1 by {(returns_model2 - returns_model1)}%")
        model_num = 2
        winner_model = model_2

    # print(f"Model 1 Correctness = {model1_correctness}")
    # print(f"Model 2 Correctness = {model2_correctness}")
    # print(f"################# Saving the Best Model = Model {model_num} ###################")
    with open(best_model_filename, 'wb') as f:
        pickle.dump(winner_model, f)
        
    
    # print(f"Model {best_model_filename} Saved!")
    return model_num

def update_list_of_models(stock_list):
    ret_list = []
    for s in stock_list:
        s = str(s).upper()
        ret_list.append(create_model(stock=s))
    
    return ret_list
    
if __name__ == "__main__":
    stock = str(input("Enter Stock Symbol : ")).upper()
    model_num = create_model(stock=stock)
    print(f"Best model = {model_num}")

