# from YT, URL: https://www.youtube.com/watch?v=9Y3yaoi9rUQ&t=1512s
# subject: Algorithmic Trading – Machine Learning & Quant Strategies Course with Python

from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta
import warnings
warnings.filterwarnings('ignore')

sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies') # note that the list may change from time to time (some small companies might be replaced)
# sp500 = pd.read_html('https://en.wikipedia.org/wiki/S%26P_500') # wrong link

# we get a list of 2 dicts, and we extract the 1st dict:
sp500 = sp500[0]

## as requested in TS = 17:56
sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
# without it, we'll get the following errors upon download data from yfinance:
# ['BF.B']: YFPricesMissingError('possibly delisted; no price data found  (1d 2017-05-12 00:00:00 -> 2025-05-10)')
# ['BRK.B']: YFTzMissingError('possibly delisted; no timezone found')




#equivalent - > list of Symbols
symbols_list = list(sp500['Symbol'].unique())
# symbols_list = sp500['Symbol'].unique().tolist()



sp500.info()
#  #   Column                 Non-Null Count  Dtype 
# ---  ------                 --------------  ----- 
#  0   Symbol                 503 non-null    object
#  1   Security               503 non-null    object
#  2   GICS Sector            503 non-null    object
#  3   GICS Sub-Industry      503 non-null    object
#  4   Headquarters Location  503 non-null    object
#  5   Date added             503 non-null    object
#  6   CIK                    503 non-null    int64 
#  7   Founded                503 non-null    object


sp500.head()
#   Symbol             Security             GICS Sector               GICS Sub-Industry    Headquarters Location  Date added      CIK      Founded
# 0    MMM                   3M             Industrials        Industrial Conglomerates    Saint Paul, Minnesota  1957-03-04    66740         1902
# 1    AOS          A. O. Smith             Industrials               Building Products     Milwaukee, Wisconsin  2017-07-26    91142         1916
# 2    ABT  Abbott Laboratories             Health Care           Health Care Equipment  North Chicago, Illinois  1957-03-04     1800         1888
# 3   ABBV               AbbVie             Health Care                   Biotechnology  North Chicago, Illinois  2012-12-31  1551152  2013 (1888)
# 4    ACN            Accenture  Information Technology  IT Consulting & Other Services          Dublin, Ireland  2011-07-06  1467373         1989



end_date = '2025-05-10'
start_date = pd.to_datetime(end_date) - pd.DateOffset(365*2) # apparently there is a limit, so I'll take only 2 years back
# start_date = pd.to_datetime(end_date) - pd.DateOffset(365*8)


df = yf.download(tickers=symbols_list, start = start_date, end = end_date, auto_adjust=False)



# df.info() # before df.stack()
# DatetimeIndex: 501 entries, 2023-05-11 to 2025-05-09
# Columns: 3018 entries, ('Adj Close', 'A') to ('Volume', 'ZTS')
# dtypes: float64(2519), int64(499)
# memory usage: 11.5 MB

df = df.stack() # looks like it was gropuing according to the left column ('dates')
# creating multi-index: 'date' + 'symbol' per date

df.index[:5]
# MultiIndex([('2023-05-11',    'A'),
#             ('2023-05-11', 'AAPL'),
#             ('2023-05-11', 'ABBV'),
#             ('2023-05-11', 'ABNB'),
#             ('2023-05-11',  'ABT')],
#            names=['Date', 'Ticker'])



df.info() # after df.stack()
# MultiIndex: 251170 entries, (Timestamp('2023-05-11 00:00:00'), 'A') to (Timestamp('2025-05-09 00:00:00'), 'ZTS')
# Data columns (total 6 columns):
#  #   Column     Non-Null Count   Dtype
# ---  ------     --------------   -----
#  0   Adj Close  251170 non-null  float64
#  1   Close      251170 non-null  float64
#  2   High       251170 non-null  float64
#  3   Low        251170 non-null  float64
#  4   Open       251170 non-null  float64
#  5   Volume     251170 non-null  float64



df.head()
# Price               Adj Close       Close        High         Low        Open      Volume                                                                                                                                              
# Date       Ticker                                                                                                                                                                                                                      
# 2023-05-11 A       125.803558  127.660004  127.699997  125.470001  127.169998   1580600.0                                                                                
#            AAPL    172.015244  173.750000  174.589996  172.169998  173.850006  49514700.0                                                                                
#            ABBV    135.933655  146.589996  147.570007  145.190002  147.289993   4300400.0                                                                                
#            ABNB    111.199997  111.199997  114.459999  111.040001  113.139999   9579400.0
#            ABT     105.743347  110.050003  110.470001  109.260002  110.059998   3879400.0


df.index.names = ['date', 'ticker']

df.columns = df.columns.str.lower()

df.head(1)
# Price               adj close       close        high         low        open     volume
# date       ticker
# 2023-05-11 A       125.803558  127.660004  127.699997  125.470001  127.169998  1580600.0



# Calculating the Garman-Klass Volatility (refer to 'Garman-Klass-Volatility.png' image)

df['garman-klass_vol'] = ((np.log(df['high']) - np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)

# calculating the 'rsi' indicator! (TBD learn this!)
# level=0 is the 'date' and level=1 is the 'ticker' (try 'apply' instead of 'transform')
# e.g. df['Age'] = df['Age'].apply(lambda x: x if x<=25 else x/10)
# transform() : Use when you need to apply a function to each column within a group and preserve the original shape of the DataFrame
df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))

df.tail()                                                                                       
# Price               adj close       close        high         low        open     volume  garman-klass_vol        rsi
# date       ticker
# 2025-05-09 XYL     122.849998  122.849998  124.000000  122.620003  123.559998   664800.0          0.000050  57.264323
#            YUM     147.130005  147.130005  148.970001  146.490005  148.259995  1068500.0          0.000118  48.007863
#            ZBH      95.209999   95.209999   96.580002   94.360001   95.080002  3438600.0          0.000270  39.772617
#            ZBRA    266.709991  266.709991  268.579987  264.640015  267.670013   700800.0          0.000104  52.322959
#            ZTS     159.270004  159.270004  162.440002  159.080002  161.410004  4243100.0          0.000150  52.741649


df.xs('AAPL', level=1)['rsi'].plot()
# plt.show() # refer to 'Apple_RSI_Plot.png'


# continue from: 
# https://www.youtube.com/watch?v=9Y3yaoi9rUQ&t=1490s