# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
# %matplotlib inline

df = pd.read_csv('StockPrices.csv')
df['Date']= pd.to_datetime(df['Date'])
df_close = df[['Date', 'Ticker', 'Close']] 
df_close.info()

#Closing Prices Dataframe
df_close = df_close.pivot_table(index = 'Date', columns = 'Ticker', values='Close').dropna(axis=1)
df_close.head()

#Calculation of the Log Returns
df_returns = (df_close.apply(lambda x: np.log(x) - np.log(x.shift(1)))).iloc[1:]
df_returns.head()

#Calculation of Moving Average for Stock Price
def dailyMovingAverage(df_close, moving_avg_period):
  dates = df_close.index[moving_avg_period:] #Storing dates of the required dates
  stocks = df_close.columns #Storing stock names
  moving_avg = []
  
  for i in range(df_close.shape[0]-moving_avg_period):
    mean = df_close.iloc[i:i+moving_avg_period,:].mean()
    moving_avg.append(mean)

  return pd.DataFrame(data = moving_avg, index = dates, columns = stocks)

movingAverage = dailyMovingAverage(df_close, 252)

movingAverage.head()

plt.plot(movingAverage.iloc[:,0], label = 'MA252')
plt.plot(df_close.iloc[:,0], label = 'Closing Price')
plt.legend()
plt.show()

## Exporting the Dataset
movingAverage.to_csv('MovingAverage.csv')

"""# Linear Regression

### PCA

First, we perform PCA on our full features dataset to feed into our linear regression.
"""

#Dataset we are compressing, column level 0 = Stock, column level 1 = feature
raw_df = df.drop(columns = ['Unnamed: 0','Close'], errors='ignore').set_index(['Date' , 'Ticker']).unstack(level = 1).stack(level = 0).unstack()
raw_df = raw_df.dropna(axis = 1)
raw_df.head()

raw_df = raw_df.to_numpy()
raw_df.shape

"""How many principal components to keep?"""

#Scaling the data
raw_df_scaled = MinMaxScaler().fit_transform(raw_df)

#Performing PCA ~ Reducing Dimensionality
PCA_model = PCA(n_components=0.95)
PCA_df = PCA_model.fit_transform(raw_df_scaled)

plt.plot(np.cumsum(PCA_model.explained_variance_ratio_))
plt.xlabel('Num Components')
plt.ylabel('Cumulative Explained Variance');

"""Storing the stock names, and dates"""

dates = df_close.index
stocks = df_close.columns
PC_labs = []
for i in range(PCA_df.shape[1]):
  lab = "PC" + str(i+1)
  PC_labs.append(lab)

"""### Linear Regression Prediction Functions"""

#Using the full features dataset, the closing prices; we are able to fit a line over a specified time period
def predict_prices(raw_df, close, time, lookback, forward, stock_num):
  
  pca = PCA(n_components=0.95)
  scaler = MinMaxScaler()

  #Training data = t - forward - lookback
  X_train = raw_df[time-forward-lookback:time-forward,:]
  X_train = scaler.fit_transform(X_train)
  X_train = pca.fit_transform(X_train)
  
  y_train = close.iloc[time-forward+1:time+1,stock_num]

  #Testing = t - lookback
  X_test = raw_df[time-lookback:time,:]
  X_test = scaler.transform(X_test)
  X_test = pca.transform(X_test)
  
  y_test = close.iloc[time+1 : time+forward+1, stock_num]

  LR = LinearRegression()
  LR.fit(X_train, y_train)
  predicted = LR.predict(X_test)

  return predicted, y_test

#This function creates the entire table of features
def construct_prediction_tab(full_features_df,closing_prices_df):
  predictions = []
  actuals = [] 
  
  for stocks in range(closing_prices_df.shape[1]):
    stock_predictions = []
    stock_actuals = []
    
    for dates in range(60, df_close.shape[0]-30, 30): 
      pred, act = predict_prices(full_features_df, closing_prices_df, dates, 30, 30, stocks)
      stock_predictions.append(pred)
      stock_actuals.append(act)

    stock_predictions = np.concatenate(stock_predictions)
    stock_actuals = np.concatenate(stock_actuals)

    predictions.append(stock_predictions)
    actuals.append(stock_actuals)

  return predictions, actuals

"""### Making Predictions"""

pred, act = construct_prediction_tab(raw_df, df_close)

"""### Creating Dataframe for Predictions and Actuals"""

# Need to get rid of 60 days for initial prediction window
final_actuals = pd.DataFrame(data = act, index=stocks, columns = dates[61:61+act[0].shape[0]]).transpose()
final_preds = pd.DataFrame(data = pred, index = stocks).transpose() 

final_preds.index = dates[61:61+pred[0].shape[0]]

final_actuals.head()
final_preds.head()

"""### Exporting the Predictions"""

final_actuals.to_csv('LR_Actual_Prices.csv')
final_preds.to_csv('LR_Predicted_Prices.csv')

"""# Diagnostics"""

# Three different Prediction Windows
p1 , t1 = predict_prices(raw_df, df_close, 60, 30, 30, 5)
p2 , t2 = predict_prices(raw_df, df_close, 90, 30, 30, 5)
p3 , t3 = predict_prices(raw_df, df_close, 120, 30, 30, 5)

predictions = np.concatenate([p1,p2,p3])
actuals = np.concatenate((t1,t2))

#This is a plt for the first 90 days of predictions for the first stock
plt.plot(predictions, label = 'predicted')
plt.plot(actuals, label = 'Actual')
plt.legend()
plt.show()

stock_predictions = []
stock_actuals = []

for i in range(60, df_close.shape[0]-30, 30):
  pred, act = predict_prices(raw_df, df_close, i, 30, 30, 5)
  stock_predictions.append(pred)
  stock_actuals.append(act)

stock_predictions = np.concatenate(stock_predictions)
stock_actuals = np.concatenate(stock_actuals)

# Q-Q plot for predictions vs actuals
plt.scatter(x = stock_predictions, y = stock_actuals)

#Full Prediction vs Actuals for the same stock
plt.figure(figsize=(20,10))
plt.plot(stock_predictions, label = 'Predicted')
plt.plot(stock_actuals, label = 'Actual')
plt.legend()

mean_absolute_error(final_actuals.dropna(), final_preds.dropna())