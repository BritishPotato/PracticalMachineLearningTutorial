# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 16:48:51 2017

@author: Denizhan Akar
"""

# Machine learning basically boils down to features (input) and labels (output)

import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt # plot stuff
from matplotlib import style # how to make it look good

style.use("ggplot") # which good plot
# preprocessing: scales your data, features into -1 to +1
# cross_validation: training/testing samples, shuffles your data, removes bias
# svm: support vector machine --> do regression
# linearRegression: regression

df = quandl.get("WIKI/GOOGL")

# print(df.head())

# You want to simplify your data as much as possible

df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]

# High and low tells us about the volatility (unpredictability, change)

# Open price vs close price shows how much it went up/down

# HL_PCT = High minus low percent
# Add column type
df["HL_PCT"] = (df["Adj. High"] - df["Adj. Low"]) / df["Adj. Low"] * 100.0
df["PCT_change"] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"]

# New df to see
df = df[["Adj. Close", "HL_PCT", "PCT_change", "Adj. Volume"]]

# Column to forecast
forecast_col = "Adj. Close"
df.fillna(-9999, inplace=True)  # Fill empty cell (NaN is not accepted)

# The actual forecast
forecast_out = int(math.ceil(0.01*len(df)))

df["label"] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

# .drop returns a new data frame, .array converts to array
X = np.array(df.drop(["label"], 1))

X=preprocessing.scale(X) 
#print("X after preprocessing.scale ",X) 
X_lately = X[-forecast_out:] 
#print("X_lately",X_lately) 
X=X[:-forecast_out] 

#print(df) 
#print("X",X) 
Y=np.array(df['label']) 
#Y=preprocessing.scale(Y) 
Y=Y[:-forecast_out] 
#print("Y ",Y)

# 20% of data, shuffle up. trains are to fit classifiers
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

# fit = train, test = score
# Made seperate because otherwise the machine will already know the answer
clf = LinearRegression(n_jobs=1)
# search for n_jobs to see how many threads for default
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
# Confidence != Accuracy

forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)
df["Forecast"] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400 # how many seconds in a day
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
    
df["Adj. Close"].plot()
df["Forecast"].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()