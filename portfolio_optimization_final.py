import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, Bounds, LinearConstraint
from numpy.linalg import norm
from datetime import timedelta
from dateutil.parser import parse
import math
import warnings
warnings.filterwarnings('ignore')

print("Loading datasets...")

# Reconstruct Closing Prices from the main dataset
df = pd.read_csv('StockPrices.csv')
df['Date'] = pd.to_datetime(df['Date'])
Closing_Prices = df.pivot_table(index='Date', columns='Ticker', values='Close').dropna(axis=1)

# Model 1: Moving Average
Moving_Average = pd.read_csv('MovingAverage.csv')
Moving_Average['Date'] = pd.to_datetime(Moving_Average['Date'])
Moving_Average = Moving_Average.set_index('Date')

# Model 2: Linear Regression
LR_Predicted_Prices = pd.read_csv('LR_Predicted_Prices.csv')
LR_Predicted_Prices['Date'] = pd.to_datetime(LR_Predicted_Prices['Date'])
LR_Predicted_Prices = LR_Predicted_Prices.set_index('Date')

LR_Actual_Prices = pd.read_csv('LR_Actual_Prices.csv')
LR_Actual_Prices['Date'] = pd.to_datetime(LR_Actual_Prices['Date'])
LR_Actual_Prices = LR_Actual_Prices.set_index('Date')

# Model 3: LSTM
PCA_Predicted_Prices = pd.read_csv('PCA_Predicted_Prices.csv')
PCA_Predicted_Prices['Date'] = pd.to_datetime(PCA_Predicted_Prices['Date'])
PCA_Predicted_Prices = PCA_Predicted_Prices.set_index('Date')

PCA_Actual_Prices = pd.read_csv('PCA_Actual_Prices.csv')
PCA_Actual_Prices['Date'] = pd.to_datetime(PCA_Actual_Prices['Date'])
PCA_Actual_Prices = PCA_Actual_Prices.set_index('Date')

print("Computing Log Returns...")
PCA_Predicted_Returns = PCA_Predicted_Prices.apply(lambda x: np.log(x) - np.log(x.shift(1))).iloc[1:] 
PCA_Actual_Returns = PCA_Actual_Prices.apply(lambda x: np.log(x) - np.log(x.shift(1))).iloc[1:] 
LR_Predicted_Returns = LR_Predicted_Prices.apply(lambda x: np.log(x) - np.log(x.shift(1))).iloc[1:] 
LR_Actual_Returns = LR_Actual_Prices.apply(lambda x: np.log(x) - np.log(x.shift(1))).iloc[1:] 
Closing_Prices_Returns = Closing_Prices.apply(lambda x: np.log(x) - np.log(x.shift(1))).iloc[1:]

# --- Helper Functions ---
def mean_returns(df, length): 
    return df.sum(axis=0) / length

def monthdelta(date, delta):
    m, y = (date.month+delta) % 12, date.year + ((date.month)+delta-1) // 12
    if not m: m = 12
    d = min(date.day, [31, 29 if y%4==0 and not y%400==0 else 28,31,30,31,30,31,31,30,31,30,31][m-1])
    new_date = (date.replace(day=d, month=m, year=y))
    return parse(new_date.strftime('%Y-%m-%d'))

def windowGenerator(dataframe, lookback, horizon, step, cummulative=False):
    if cummulative:
        c = lookback
        step = horizon
        
    initial = min(dataframe.index)
    windows = []
    horizons = []

    while initial <= monthdelta(max(dataframe.index), -lookback):
        windowStart = initial
        windowEnd = monthdelta(windowStart, lookback)
        if cummulative:
            windowStart = min(dataframe.index)
            windowEnd = monthdelta(windowStart, c) + timedelta(days=1)
            c += horizon
        horizonStart = windowEnd + timedelta(days=1)
        horizonEnd = monthdelta(horizonStart, horizon)

        windows.append(dataframe[windowStart:windowEnd])
        horizons.append(dataframe[horizonStart:horizonEnd])
        initial = monthdelta(initial, step)

    return windows, horizons

def actual_return(actual_returns, w):
    mean_return = mean_returns(actual_returns, actual_returns.shape[0])
    actual_covariance = actual_returns.cov()
    portfolio_returns = mean_return.T.dot(w)
    portfolio_variance = w.T.dot(actual_covariance).dot(w)
    return portfolio_returns, portfolio_variance

def scipy_opt(predicted_returns, actual_returns, lam1, lam2):
    mean_return = mean_returns(predicted_returns, predicted_returns.shape[0])
    predicted_covariance = predicted_returns.cov()
    
    def f(w):
        return -(mean_return.T.dot(w) - lam1*(w.T.dot(predicted_covariance).dot(w)) + lam2*norm(w, ord=1))

    opt_bounds = Bounds(0, 1)
    cons = ({'type': 'eq', 'fun': lambda w: sum(w) - 1})

    sol = minimize(f, x0=np.ones(mean_return.shape[0])/mean_return.shape[0], constraints=cons, bounds=opt_bounds, options={'disp': False}, tol=10e-10)

    w = sol.x
    predicted_portfolio_returns = w.dot(mean_return)
    portfolio_STD = w.T.dot(predicted_covariance).dot(w)
    
    portfolio_actual_returns, portfolio_actual_variance = actual_return(actual_returns, w)
    
    # Avoid divide by zero
    std_dev = np.std(portfolio_actual_variance)
    sharpe_ratio = portfolio_actual_returns / std_dev if std_dev != 0 else 0

    return {
        'weights': w,
        'actual_returns': portfolio_actual_returns,
        'actual_variance': portfolio_actual_variance,
        'sharpe_ratio': sharpe_ratio
    }

# --- Execution ---
print("Running SciPy Optimizer...")
lookback = 12

MA_act_windows, MA_act_horizons = windowGenerator(Closing_Prices_Returns, lookback, 1, 1)
LR_pred_windows, LR_pred_horizons = windowGenerator(LR_Predicted_Returns, lookback, 1, 1)
LR_act_windows, LR_act_horizons = windowGenerator(LR_Actual_Returns, lookback, 1, 1)
LSTM_pred_windows, LSTM_pred_horizons = windowGenerator(PCA_Predicted_Returns, lookback, 1, 1)
LSTM_act_windows, LSTM_act_horizons = windowGenerator(PCA_Actual_Returns, lookback, 1, 1)

# Ensure uniform testing window size across all models (5 years / 60 months max)
test_months = min(60, len(MA_act_horizons), len(LR_act_horizons), len(LSTM_act_horizons))
start_idx = -test_months

MA_returns, LR_returns, LSTM_returns = [], [], []
timestamps = []

for i in range(len(LSTM_act_horizons) + start_idx, len(LSTM_act_horizons)):
    ma_opt = scipy_opt(MA_act_windows[i], MA_act_horizons[i], .5, 2)
    MA_returns.append(ma_opt['actual_returns'])
    
    lr_opt = scipy_opt(LR_pred_horizons[i], LR_act_horizons[i], .5, 2)
    LR_returns.append(lr_opt['actual_returns'])
    
    lstm_opt = scipy_opt(LSTM_pred_horizons[i], LSTM_act_horizons[i], .5, 2)
    LSTM_returns.append(lstm_opt['actual_returns'])
    
    timestamps.append(LSTM_act_horizons[i].index[0])

# --- Diagnostics & Plotting ---
print("Generating Equity Curves...")
MA_equity, LR_equity, LSTM_equity = [100], [100], [100]

for i in range(1, test_months):
    MA_equity.append(MA_equity[i-1] * math.exp(MA_returns[i]))
    LR_equity.append(LR_equity[i-1] * math.exp(LR_returns[i]))
    LSTM_equity.append(LSTM_equity[i-1] * math.exp(LSTM_returns[i]))

plt.figure(figsize=(12, 6))
plt.plot(timestamps, MA_equity, label="Moving Average", color='red')
plt.plot(timestamps, LR_equity, label="Linear Regression", color='green')
plt.plot(timestamps, LSTM_equity, label="LSTM", color='blue')
plt.title("Equity Graph: 97 Indian Stocks Portfolio")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid(True)
plt.show();

print("--- Final Equity ---")
print(f"Moving Average Ending Equity: ${MA_equity[-1]:.2f}")
print(f"Linear Regression Ending Equity: ${LR_equity[-1]:.2f}")
print(f"LSTM Ending Equity: ${LSTM_equity[-1]:.2f}")

def metrics(returns): 
    sharpe = returns.mean() / returns.std()
    annualized_sharpe = sharpe.item() / math.sqrt(252)
    annualized_vol = returns.std().item() / math.sqrt(252)
    return {"Annualized Sharpe Ratio": round(annualized_sharpe, 4), "Annualized Volatility": round(annualized_vol, 4)}

print("\n--- Metrics ---")
print("Moving Average:", metrics(np.array(MA_returns)))
print("Linear Regression:", metrics(np.array(LR_returns)))
print("LSTM:", metrics(np.array(LSTM_returns)))