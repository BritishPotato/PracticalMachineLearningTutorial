# Machine learning basically boils down to features (input) and labels (output)

import pandas as pd
import quandl
import math
import numpy as np
# from sklearn import preprocessing, cross_validation, svm
# from sklearn.linear_model import LinearRegression
# preprocessing: scales your data, features into -1 to +1
# cross_validation: training/testing samples, shuffles your data, removes bias
# svm: support vector machine --> do regression
# linearRegression: regression

df = quandl.get("WIKI/GOOGL")

# print(df.head())

# You want to simplify your data as much as possible

df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume",]]

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
print(df.head())

# .drop returns a new data frame, .array converts to array
x = np.array(df.drop(["label"],1))
y = np.array(df["label"])

# Scales x before feeding it through classifier. If doing high frequency, skip.
# x = preprocessing.scale(x)

x = x[:-forecast_out+1]
df.dropna(inplace=True)
y = np.array(df["label"])

print(len(x), len(y))
