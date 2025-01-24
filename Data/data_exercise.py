import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics

import statsmodels.tsa.stattools as ts
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

pd.set_option('display.max_rows', None)  # None means unlimited
pd.set_option('display.max_columns', None)

# As always, first thing to do is get a sense of what the stock data looks like. I put
# a large portion of the scratch code involved in exploring the data here:
def exploredf(df):
    print(df.head(10))
    print(df.describe())
    # Notes:
    # 1. The day and timestr are separate columns, so there's a multi-level indexing of the data
    # 2. The minima for (a) and (d) are orders of magnitude lower than their means, which is not true for the other columns
    # and those minima differ strongly from the 25%...
    # So probably there is bad data there that needs to be cleaned. (Unsurprising, given instructions)

    # Okay, while we're checking for things to clean, we should check for NaNs:
    print(df.isna().sum())

    # If we treat anomalous values of (a) and (d) as NaNs, how does this balance change?

    # Considering the 25% mark for a and d are around 341 and 46 respectively,
    # and c has higher std than d but a minimum of 33 (and a similar-order magnitude mean),
    # would expect any prices below 30 are anomalous, garbage data:
    threshold = 2
    df_copy = df.copy()
    for i in ['a', 'b', 'c', 'd', 'e', 'f']:
        df_copy.loc[df_copy[i] < threshold, i] = None
    print(df_copy.isna().count())

    # So with the exception of f, less than a day's worth of data is missing from the other columns
    # Are there any patterns about where the missing data is located?

    for i in ['a', 'c', 'd', 'f']:
        nan_counts = df_copy[df_copy[i].isna()].groupby('day').size()
        nan_counts.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.ylabel('Number of NaNs')
        plt.title('Number of NaNs in column a by Day')
        plt.show()

    # a's NaNs: Mostly spread out, but many on day 79 and 199
    # c's NaNs: Sufficiently spread out; most on 166 and 171
    # d's NaNs: Very spread out; most on 117 and 171
    # f's NaNs: Strongly clustered on a few days. Significant NaNs on:
    # Days 5, 8*, 121, 129, 141, 260, 261, 302, 360

    # Does anything significant happen on these days?
    for i in ['a', 'b', 'c', 'd', 'e', 'f']:
        df.plot(x='day', y=i, kind='scatter')
        plt.show()
    # Based on visual inspection, for everything except for f, not really.
    # For f, it does seem like there is more notable activity on some of the
    # days, but it doesn't appear as though there is any particular correlation.

# Simplest solution: Forward-fill the missing data. The effect should ...
# be that volatility in these periods is slightly under-estimated
# but I can change my decision later if necessary...
def clean_data(df):
    df_copy = df.copy()
    threshold1 = 2
    threshold2 = 0.1
    for i in ['a', 'b', 'c', 'd', 'e', 'f']:
        # Replace NaNs and below-threshold values via forward fill
        df_copy.loc[df_copy[i] < threshold1, i] = None
        df_copy[i].fillna(method='ffill', inplace=True)

        # Replace other extreme values with rolling median:
        relative_threshold = threshold2 * np.log(df_copy[i][0])
        df_copy['smoothed'] = df_copy[i].rolling(window=8,center=True).median()
        # counter = 0
        # for j in range(0,len(df_copy[i])):
        #     if abs(df_copy[i][j] - df_copy['smoothed'][j]) > relative_threshold:
        #         counter += 1
        # print(i, counter)
        df_copy[i] = np.where(abs(df_copy[i] - df_copy['smoothed']) > relative_threshold, df_copy['smoothed'], df_copy[i])
    return df_copy

# The following determines the high, low, open, and close on a period of 'days' days:
# The values returned are not directly the high, low, open, and close, but are related...
# and are the quantities used in the volatility calculations presented in Yang and Zhang
def days_data(df,days):
    # Following notation in Yang and Zhang:
    O1 = df.groupby('day').first().reset_index()
    O1 = O1[['a','b','c','d','e','f']].groupby(O1.index // days).first()
    H1 = df.groupby('day').max().reset_index()
    H1 = H1[['a','b','c','d','e','f']].groupby(H1.index // days).max()
    L1 = df.groupby('day').min().reset_index()
    L1 = L1[['a','b','c','d','e','f']].groupby(L1.index // days).min()
    C1 = df.groupby('day').last().reset_index()
    C1 = C1[['a','b','c','d','e','f']].groupby(C1.index // days).last()
    C0 = C1.shift(1)

    C0.fillna(method='bfill', inplace=True)
    o = log_of_df(O1) - log_of_df(C0)
    u = log_of_df(H1) - log_of_df(O1)
    d = log_of_df(L1) - log_of_df(O1)
    c = log_of_df(C1) - log_of_df(O1)

    return o, u, d, c

# The following is a convenient function for computing the log of all stock values in the dataframe:
def log_of_df(df):
    logdf = df[['a','b','c','d','e','f']].copy()
    for col in ['a', 'b', 'c', 'd', 'e', 'f']:
        logdf[col] = np.log(df[col])
    return logdf

# The following contains methods to calculate the 5 different volatility metrics presented in Yang and Zhang:

class DayVolatility:
    # Again, following notation in Yang and Zhang:
    def __init__(self, o, u, d, c, w):
        self.o = o
        self.u = u
        self.d = d
        self.c = c
        self.n = len(o)
        self.w = w

    def drift(self):
        return (self.o + self.c).rolling(window=self.w).mean().dropna()

    def closeclose(self):
        return (self.n**0.5) * ((self.o + self.c).rolling(window=self.w).std() ).dropna()

    def parkinson(self):
        inner_term = ((self.u - self.d) ** 2).rolling(window=self.w).sum()
        scaling_factor = 1 / (self.w * 4 * np.log(2))
        return (self.n**0.5) * ((scaling_factor * inner_term) ** 0.5).dropna()

    def garmanklass(self):
        first_term = 0.5 * (self.u - self.d) ** 2
        second_term = (2 * np.log(2) - 1) * self.c ** 2
        rolling_sum = (first_term - second_term).rolling(window=self.w).sum()
        return (self.n**0.5) * (((1/self.w) * rolling_sum)**0.5).dropna()

    def rogersatchell(self):
        first_term = (self.u * (self.u - self.c)).rolling(window=self.w).sum()
        second_term = (self.d * (self.d - self.c)).rolling(window=self.w).sum()
        return (self.n**0.5) * (((1 / self.w) * (first_term + second_term)) ** 0.5).dropna()

    def yangzhang(self):
        rs = self.rogersatchell() / (self.n**0.5)
        VO = self.o.rolling(window=self.w).var().dropna()
        VC = self.c.rolling(window=self.w).var().dropna()
        k = 0.34 / (1.34 + (self.n + 1)/(self.n - 1))
        return (self.n**0.5) * (VO + k*VC + (1-k)*(rs**2))**0.5

# The following was my attempt to run a linear regression between a stock's current volatility and its future
# volatility, similar to what was attempted by mrichman:
def regression(column, w, end):
    # Train on the third period ago vs the second period ago,
    # test on the second period ago vs the most recent period:
    x = column[end-(3*w):end-(2*w)].values.reshape(-1,1)
    y = column[end-(2*w):end-w].values
    testing = column[end-w:end].values.reshape(-1,1)
    model = LinearRegression()
    model.fit(x,y)

    # Find the slope and intercept of the line of best fit:
    slope = model.coef_[0]
    intercept = model.intercept_
    print(f"Slope (Coefficient): {slope}")
    print(f"Intercept: {intercept}")
    predictions = model.predict(testing)

    # Mean Squared Error
    mse = metrics.mean_squared_error(testing, predictions)
    print(f"Mean Squared Error (MSE): {mse}")

    # R^2 Score
    r2 = metrics.r2_score(testing, predictions)
    print(f"R^2 Score: {r2}")

    # Correlation
    correlation_matrix = np.corrcoef(testing.flatten(), predictions)
    correlation = correlation_matrix[0, 1]
    print(f"Pearson Correlation Coefficient: {correlation}")

# I didn't really have a plan for this; more just curios about the relationships between stocks:
def stock_linear_regressions(df, splitnum):
    l = len(df)
    lst = ['a', 'b', 'c', 'd', 'e', 'f']
    lst2 = lst.copy()
    for col in lst:
        lst2.remove(col)
        for col2 in lst2:
            # Train on the second most recent period, test on the most recent period:
            trainx = df[col][(splitnum - 2) * (l // splitnum):(splitnum-1) * (l // splitnum)].values.reshape(-1, 1)
            trainy = df[col2][(splitnum - 2) * (l // splitnum):(splitnum-1) * (l // splitnum)].values
            testx = df[col][(splitnum - 1) * (l // splitnum):l].values.reshape(-1, 1)
            testy = df[col2][(splitnum - 1) * (l // splitnum):l].values

            # Fit a linear regression model:
            model = LinearRegression()
            model.fit(trainx, trainy)
            slope = model.coef_[0]
            intercept = model.intercept_
            predictions = testx * slope + intercept

            # Filter the test values and predictions for anomalies:
            differences = np.abs(testy - predictions.flatten())
            threshold = np.percentile(differences, 99)
            mask = differences < threshold
            ftesty = testy[mask]
            fpreds = predictions.flatten()[mask]

            # Retrieve metrics about goodness-of-fit:
            mse = metrics.mean_squared_error(ftesty, fpreds)
            r2 = metrics.r2_score(ftesty, fpreds)
            correlation_matrix = np.corrcoef(ftesty, fpreds)
            correlation = correlation_matrix[0, 1]

            # If the fit is reasonably good, tell me about it:
            if r2 > 0 and correlation > 0.5:
                print(col, col2, ': mse: ', mse, ', r2: ', r2, ', corl: ', correlation)
                plt.scatter(predictions.flatten(), testy)
                plt.show()

# The following was my attempt at analyzing the autoregressive behavior of
# stock prices, log returns, and volatility estimates:
def timeseriesanalyze(df):
    for col in ['a','b','c','d','e','f']:
        # Apply the ADF test:
        print(ts.adfuller(df[col]))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        numlags = min(len(df) / 4, 60)

        # Show the autocorrelation function plot
        plot_acf(df[col], lags=numlags, ax=axes[0])

        # Show the partial autocorrelation function plot
        plot_pacf(df[col], lags=numlags, ax=axes[1])

        plt.tight_layout()
        plt.show()

# If my data made it through the above function, I would input it into this:
def timeseriesmodel(df):
    for col in ['a','b','c','d','e','f']:
        # Split into training and testing portions
        # (yes I know I've been inconsistent about how I split the data up; apologies...
        # I ran out of time to clean things up more)
        train_size = int(len(df) * 0.5)
        train, test = df[col][:train_size], df[col][train_size:]

        # Train the model:
        model = ARIMA(train, order=(1,1,2))
        result = model.fit()

        # This is supposed to tell you how good the model is; I don't really understand how it works:
        # (but the number sure was big for everything I tried!)
        aic = result.aic
        print(aic)

        # Plot the residuals to see if any autocorrelation wasn't accounted for:
        residuals = result.resid
        # residuals.plot()
        # plt.show()
        #
        # plot_acf(residuals, lags=10)
        # plt.show()

        # Compare predictions to the result:
        prediction = result.forecast(steps=len(test))
        print(prediction.mean())
        print(test.mean())
        #plt.plot(forecast,test)
        #plt.show()

        # Most of the plots I looked at here looked more like a child's scribble than
        # any sort of nice correlation...

# What's the best window for my volatility estimate, if I am just going to use the last row?
# Restricting to values divisible by 4 or 6 so everything rounds more nicely...
# Note the window is len(daily data) / i, so larger values of i = shorter window
# Goodness of window determined by having the highest r2 score (none of the r2 scores are particularly good)
# (I'm not sure if this is the best way to compare windows,
# but I ran out of time to find a better method)
def bestwindow(o, u, d, c):
    # Initialize a dictionary of windows and r2 scores
    bestpredictor = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 'f': 0}
    bestval = {'a': -60000000000, 'b': -600000000, 'c': -600000000, 'd': -600000000, 'e': -600000000, 'f': -60000000}
    mse = {'a': -60000000000, 'b': -600000000, 'c': -600000000, 'd': -600000000, 'e': -600000000, 'f': -60000000}
    # Run over 2 months window to 1 week window:
    for i in range(6,53):
        # If the window is "nice" (divisible by 4 or 6 so everything rounds well)...
        if (i % 6 == 0 or i % 4 == 0):
            # Get the rolling volatility window
            vals = DayVolatility(o, u, d, c, len(o) // i).yangzhang()
            # For each stock,
            for col in ['a','b','c','d','e','f']:
                # Determine how good the volatility estimate is at predicting
                # the volatility in one month, compared to the best result found so far:
                if (metrics.r2_score(vals[col].iloc[(8*len(o) // 12):], vals[col].shift((len(o) // 12)).iloc[(8*len(o) // 12):])) > bestval[col]:
                    # If a new best is found, set as such, modify the best r2 found and MSE accordingly:
                    bestpredictor[col] = i
                    bestval[col] = metrics.r2_score(vals[col].iloc[(8*len(o) // 12):], vals[col].shift((len(o) // 12)).iloc[(8*len(o) // 12):])
                    mse[col] = metrics.mean_squared_error(vals[col].iloc[(8*len(o) // 12):], vals[col].shift((len(o) // 12)).iloc[(8*len(o) // 12):])
    print(bestpredictor)
    print(mse)
    # Result found:
    # a: 8, b: 52, c: 12, d: 8, e: 6, f: 6

    # What is the std of my volatility estimate over a month?
    # print('yz52', DayVolatility(o, u, d, c, len(o)//52).yangzhang().tail(21).std())
    # print('yz12', DayVolatility(o, u, d, c, len(o)//12).yangzhang().tail(21).std())
    # print('yz8', DayVolatility(o, u, d, c, len(o)//8).yangzhang().tail(21).std())
    # print('yz6', DayVolatility(o, u, d, c, len(o)//6).yangzhang().tail(21).std())
    # Might be a good alternative metric to MSE for evaluating predictive ability of
    # volatility estimate...


# Here's where I played around with the functions above and imported the data:
def main():
    # Read the stock data as a dataframe, since pandas is what I have familiarity with:
    df = pd.read_csv('stockdata3.csv')

    # Clean the dataframe, take the log:
    clean_df = clean_data(df)
    logdf = log_of_df(clean_df)

    # Per Yang-Zhang's notation, the summary data for each day
    # (related to high, low, open, close):
    o, u, d, c = days_data(clean_df,1)

    #stock_linear_regressions(logdf, 12)

    # What different values in volatility estimates do we see when measuring volatility differently?
    # print('cc', DayVolatility(o, u, d, c, len(o)//12).closeclose().tail(1))
    # print('p', DayVolatility(o, u, d, c, len(o)//12).parkinson().tail(1))
    # print('gk', DayVolatility(o, u, d, c, len(o)//12).garmanklass().tail(1))
    # print('rs', DayVolatility(o, u, d, c, len(o)//12).rogersatchell().tail(1))
    # print('yz52', DayVolatility(o, u, d, c, len(o)//52).yangzhang().tail(1))
    # print('yz24', DayVolatility(o, u, d, c, len(o)//24).yangzhang().tail(1))
    print('yz12', DayVolatility(o, u, d, c, len(o)//12).yangzhang().tail(40))
    # print('yz8', DayVolatility(o, u, d, c, len(o)//8).yangzhang().tail(1))
    # print('yz6', DayVolatility(o, u, d, c, len(o)//6).yangzhang().tail(1))
    # print('yz3', DayVolatility(o, u, d, c, len(o)//3).yangzhang().tail(1))



    # Some of the scratch code for work done to train and test ARIMA, which ultimately failed:
    # (note this does not demonstrate everything I tried to run ARIMA on; see document for details)
    #
    # voldf = DayVolatility(o,u,d,c,len(o)//12).closeclose()
    # #print(voldf)
    # regression(voldf['f'],len(o)//12,len(voldf))
    #
    # minute_returns = logdf - logdf.shift(1).fillna(method='bfill')
    # minute_returns['day'] = clean_df['day']
    # day_vol = minute_returns.groupby('day').std().dropna()
    # dayvoldiff = day_vol - day_vol.shift(1).fillna(method='bfill')
    #
    # minute_returns.plot()
    # plt.show()
    # voldiff = voldf - voldf.shift(1).fillna(method='bfill')
    # timeseriesanalyze(dayvoldiff.tail(60))
    # timeseriesmodel(dayvoldiff.tail(40))



if __name__ == "__main__": main()

