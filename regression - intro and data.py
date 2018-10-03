# Machine learning basically boils down to features (input) and labels (output)

import pandas as pd
import quandl
import math
import numpy as np

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

print(len(df))
