import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings('ignore')

# 1. Data Pulling & Setup
print("Loading StockPrices.csv for LSTM...")
df = pd.read_csv('StockPrices.csv')
df['Date'] = pd.to_datetime(df['Date'])
df_close = df[['Date', 'Ticker', 'Close']] 
df_close = df_close.pivot_table(index='Date', columns='Ticker', values='Close').dropna(axis=1)

dates = df_close.index
stocks = df_close.columns
stocks_tickers = df_close.columns

# 2. Feature Engineering
print("Engineering rolling features...")
raw_df = df.drop(columns=['Unnamed: 0'], errors='ignore').set_index(['Date', 'Ticker']).unstack(level=1).stack(level=0).unstack()
raw_df = raw_df.dropna(axis=1)

for stock in stocks_tickers:
    raw_df.loc[:, (stock, 'DailyRet')] = raw_df[stock]["Close"].pct_change()
    raw_df.loc[:, (stock, '20DayRet')] = raw_df[stock]["Close"].pct_change(20)
    roller = raw_df[stock]["DailyRet"].rolling(20)
    raw_df.loc[:, (stock, '20DayVol')] = roller.std(ddof=0)
    
    rolling_year_ret = raw_df[stock]["20DayRet"].rolling(252)
    raw_df.loc[:, (stock, 'Z20DayRet')] = (rolling_year_ret.mean().shift(1) - raw_df[stock]['20DayRet']) / rolling_year_ret.std(ddof=0).shift(1)
    
    rolling_year_vol = raw_df[stock]["20DayVol"].rolling(252)
    raw_df.loc[:, (stock, 'Z20DayVol')] = (rolling_year_vol.mean().shift(1) - raw_df[stock]['20DayVol']) / rolling_year_vol.std(ddof=0).shift(1) 

full_feature_dataset = raw_df.dropna(axis=0)
full_features_np = full_feature_dataset.to_numpy()

# 3. Data Manipulation
print("Preparing datasets for training...")
closing_prices = df_close.iloc[-full_features_np.shape[0]:, :]

array_train, array_test = train_test_split(closing_prices, shuffle=False, test_size=.2)
PCA_train_raw, PCA_test_raw = train_test_split(full_features_np, shuffle=False, test_size=.2)

# Scaling Target
scl = MinMaxScaler()
scale = MinMaxScaler()
array_train = scl.fit_transform(array_train)
array_test  = scale.fit_transform(array_test)

# Scaling & PCA for Features (Dynamic)
# FIX: We use only ONE scaler and ONE PCA model to prevent data leakage
pc_scl = MinMaxScaler()
pca = PCA(n_components=0.95)

# Fit and transform ONLY on training data
PCA_train = pc_scl.fit_transform(PCA_train_raw)
PCA_train = pca.fit_transform(PCA_train)

# Transform test data using the fitted training objects
PCA_test = pc_scl.transform(PCA_test_raw)
PCA_test = pca.transform(PCA_test) 

num_features_selected = PCA_train.shape[1]
print(f"PCA reduced features to: {num_features_selected} components")

# 4. Processing Data into Timesteps
def processData(data, lookback, horizon, num_companies, jump=1):
    X, Y = [], []
    for i in range(0, len(data) - lookback - horizon + 1, jump):
        X.append(data[i:(i+lookback)])
        Y.append(data[(i+lookback):(i+lookback+horizon)])
    return np.array(X), np.array(Y)

num_companies = df_close.shape[1]
lookback = 252 
horizon = 22 

X_PCA, void = processData(PCA_train, lookback, horizon, num_companies)
void, y = processData(array_train, lookback, horizon, num_companies)
y = np.array([list(x.ravel()) for x in y])

X_train, X_validate, y_train, y_validate = train_test_split(X_PCA, y, test_size=0.20, random_state=1)

# 5. Model Architecture + Training
print("Building and training LSTM Model...")
num_neurons_L1 = 800
num_neurons_L2 = 600
EPOCHS = 100 # Adjust this down to 20 or 30 if you want it to train faster for testing

model = Sequential()
# Input shape now dynamically maps to the PCA components kept
model.add(LSTM(num_neurons_L1, input_shape=(lookback, num_features_selected), return_sequences=True))
model.add(LSTM(num_neurons_L2, return_sequences=False)) # Fixed layer transition
model.add(Dense(horizon * num_companies, activation='relu'))
model.add(Dense(horizon * num_companies, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_validate, y_validate), shuffle=False, batch_size=256, verbose=2)

# 6. Full Data Predictions
print("Generating final predictions...")
full_PCA = np.concatenate((PCA_train, PCA_test), axis=0)

X_all, void = processData(full_PCA, lookback, horizon, num_companies, horizon)
void, y_all = processData(scl.fit_transform(df_close.iloc[-full_PCA.shape[0]:]), lookback, horizon, num_companies, horizon)
y_all = np.array([list(a.ravel()) for a in y_all])

Xt = model.predict(X_all)

# Un-scale Data
def do_inverse_transform(output_result, num_companies):
    original_matrix_format = []
    for result in output_result:
        original_matrix_format.append(scl.inverse_transform([result[x:x+num_companies] for x in range(0, len(result), num_companies)]))
    original_matrix_format = np.array(original_matrix_format)
    for i in range(len(original_matrix_format)):
        output_result[i] = original_matrix_format[i].ravel()
    return output_result

def prediction_by_step_by_company(raw_model_output, num_companies):
    matrix_prediction = []
    for i in range(0, num_companies):
        matrix_prediction.append([[lista[j] for j in range(i, len(lista), num_companies)] for lista in raw_model_output])
    return np.array(matrix_prediction)

def target_by_company(raw_model_output, num_companies):
    matrix_target = [[] for x in range(num_companies)]
    for output in raw_model_output:
        for i in range(num_companies):
            for j in range(0, len(output), num_companies):
                matrix_target[i].append(output[i+j])
    return np.array(matrix_target)

Xt = do_inverse_transform(Xt, num_companies)
predictions = prediction_by_step_by_company(Xt, num_companies)
Yt = do_inverse_transform(y_all, num_companies)
actuals = target_by_company(Yt, num_companies)

# Format Final Output Arrays
predicted_prices = np.zeros((predictions.shape[1] * predictions.shape[2], predictions.shape[0]))
for i in range(predictions.shape[0]):
    counter = 0
    for j in range(predictions.shape[1]):
        for z in range(predictions.shape[2]):
            predicted_prices[counter, i] = predictions[i, j, z]
            counter += 1

actuals_prices = actuals[:, :predicted_prices.shape[0]].T

# Create dynamic DataFrames
actual_prices = pd.DataFrame(data=actuals_prices, columns=stocks)
predicted_prices = pd.DataFrame(data=predicted_prices, columns=stocks)

# Assign dates cleanly to the end of the timeline
actual_prices.index = dates[-actual_prices.shape[0]:]
predicted_prices.index = dates[-predicted_prices.shape[0]:]

# 7. Export the Predictions
actual_prices.to_csv('PCA_Actual_Prices.csv')
predicted_prices.to_csv('PCA_Predicted_Prices.csv')
print("Model 3 execution complete! Saved PCA_Actual_Prices.csv and PCA_Predicted_Prices.csv")