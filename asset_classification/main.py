# import torch
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta  
from sklearn.model_selection import train_test_split # To split the data into training and testing sets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report #To evaluate the trained model

# print("Everything works")

#symbols selected from my investment portfolio
symbols = ["AAPL", "BAC", "EQIX", "NVDA", "PEP", "PFE", "WBD", "XOM"]

#price over the last 2 weeks, starting from today minus 90 days. 
end_date = datetime.today()
start_date = end_date - timedelta(days=90)

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

labels = returns.applymap(label_return)  #.applymap could go away in the future, I just have to switch it to .map

# Split data into training and test sets.
for symbol in symbols:
    print(f"Model for {symbol}")
    X = returns[[symbol]]
    y = labels[symbol] 
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, shuffle=False) #Test size is 30%

    #Trained the model using logistic regression
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    #Predict
    y_pred = log_reg.predict(X_test)

    #most recent return value
    latest_returns = X.iloc[[-1]]

    #Predict tomorrow's movement
    predictions = log_reg.predict(latest_returns)

    #Evaluation
    print(f"Prediction for tomorrow: {predictions}")
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

# print(y_pred)
# print(X_train)
# print(returns.head())
# print(labels.head())
# print(returns.round(2).head())
# print(data[['Adj Close', 'Daily Return']].head())
# print(data.columns)