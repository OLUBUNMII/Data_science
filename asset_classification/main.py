import torch
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# print("Everything works")

#symbols selected from my investment portfolio
symbols = ["AAPL", "BAC", "EQIX", "NVDA", "PEP", "PFE", "WBD", "XOM"]

#price over the last 2 weeks, starting from today minus 14 days. 
end_date = datetime.today()
start_date = end_date - timedelta(days=14)

#download the data from yahoo finance
data = yf.download(symbols, start=start_date, end=end_date, auto_adjust=True)

#Calculate daily rerturn. 
# NB: Adj close accounts for dividends and splits
# data['Daily Return'] = data['Adj Close'].pct_change() * 100

returns = data['Close'].pct_change().dropna() * 100

#labelling the returns with thresholds
def label_return(x):
    if x > 1:
        return 1
    elif x < -1:
        return -1
    else:
        return 0

labels = returns.applymap(label_return)
print(returns.head())
print(labels.head())
# print(returns.round(2).head())
# print(data[['Adj Close', 'Daily Return']].head())
# print(data.columns)