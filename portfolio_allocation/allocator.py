import yfinance as yf
import math 
import pandas as pd 
from config import symbols, weights

budget = float(input("Enter your total budget: $"))
data = yf.download(symbols, period="1d")["Close"]
price = data.iloc[-1] #most recent closing price of market

#function to allocate portfolio
# def allocate_portfolio(symbols, weights, budget):
portfolio = {} #to store allocation results

# for symbol in symbols:
for symbol, weight, price in zip(symbols, weights, price):

    allocation = weights[symbol] * budget #multiply the budget by the weight I set to know how much should be allocated. Everything equals to 1

    shares = math.floor(allocation / price) #calculates how much stock I can afford

    portfolio[symbol] = {
        "price": round(price, 2),              #last price, rounded
        "weight": (weights[symbol]),           #weight set
        "allocation": round(allocation, 2),    #Dollar amount allocated
        "shares_to_buy": shares                #Number of whole shares
    }

    # return portfolio #returns full breakdown

df_portfolio = pd.DataFrame(portfolio).T
print(df_portfolio)