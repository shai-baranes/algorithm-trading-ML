# from YT, URL: https://www.youtube.com/watch?v=9Y3yaoi9rUQ&t=1512s
# subject: Algorithmic Trading – Machine Learning & Quant Strategies Course with Python


# pd.set_option('display.max_rows', 0)
# pd.set_option('display.max_columns', None)
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
NUM_OF_YEARS = 5

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
start_date = pd.to_datetime(end_date) - pd.DateOffset(365*NUM_OF_YEARS) # apparently there is a limit, so I'll take only 2 years back
# start_date = pd.to_datetime(end_date) - pd.DateOffset(365*8)


df = yf.download(tickers=symbols_list, start = start_date, end = end_date, auto_adjust=False)
df.index # (the dates)
# DatetimeIndex(['2023-05-11', '2023-05-12', '2023-05-15', '2023-05-16',
#                '2023-05-17', '2023-05-18', '2023-05-19', '2023-05-22',
#                '2023-05-23', '2023-05-24',
#                ...
#                '2025-04-28', '2025-04-29', '2025-04-30', '2025-05-01',
#                '2025-05-02', '2025-05-05', '2025-05-06', '2025-05-07',
#                '2025-05-08', '2025-05-09'],
#               dtype='datetime64[ns]', name='Date', length=501, freq=None)





df.columns
# TBD - check also df.info()
# MultiIndex([('Adj Close',    'A'),
#             ('Adj Close', 'AAPL'),
#             ('Adj Close', 'ABBV'),
#             ('Adj Close', 'ABNB'),
#             ('Adj Close',  'ABT'),
#             ('Adj Close', 'ACGL'),
#             ('Adj Close',  'ACN'),
#             ('Adj Close', 'ADBE'),
#             ('Adj Close',  'ADI'),
#             ('Adj Close',  'ADM'),
#             ...
#             (   'Volume',  'WTW'),
#             (   'Volume',   'WY'),
#             (   'Volume', 'WYNN'),
#             (   'Volume',  'XEL'),
#             (   'Volume',  'XOM'),
#             (   'Volume',  'XYL'),
#             (   'Volume',  'YUM'),
#             (   'Volume',  'ZBH'),
#             (   'Volume', 'ZBRA'),
#             (   'Volume',  'ZTS')],
#            names=['Price', 'Ticker'], length=3018)



# df.info() # before df.stack()
# DatetimeIndex: 501 entries, 2023-05-11 to 2025-05-09
# Columns: 3018 entries, ('Adj Close', 'A') to ('Volume', 'ZTS')
# dtypes: float64(2519), int64(499)
# memory usage: 11.5 MB

# stack is applied only on the index (in this case 'date' & 'symbol')
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


df.index.names = ['date', 'ticker'] # already exists, now only to make it lower()

df.columns = df.columns.str.lower() # same here

df.head(1)
# Price               adj close       close        high         low        open     volume
# date       ticker
# 2023-05-11 A       125.803558  127.660004  127.699997  125.470001  127.169998  1580600.0



# Calculating the Garman-Klass Volatility (refer to 'Garman-Klass-Volatility.png' image)
# TBD learn more about it from the internet!
df['garman-klass_vol'] = ((np.log(df['high']) - np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)

# calculating the 'rsi' indicator! (TBD learn this!)
# note that it is the only indicator that is not been normalized (compared to the following added indicators)
# e.g. df['Age'] = df['Age'].apply(lambda x: x if x<=25 else x/10)
# transform() : Use when you need to apply a function to each column within a group and preserve the original shape of the DataFrame
df['rsi'] = df.groupby('ticker')['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))
# df['rsi'] = df.groupby('level=1')['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))


import pdb; pdb.set_trace()  # breakpoint 7065e700 //



df.tail()                                                                                       
# Price               adj close       close        high         low        open     volume  garman-klass_vol        rsi
# date       ticker
# 2025-05-09 XYL     122.849998  122.849998  124.000000  122.620003  123.559998   664800.0          0.000050  57.264323
#            YUM     147.130005  147.130005  148.970001  146.490005  148.259995  1068500.0          0.000118  48.007863
#            ZBH      95.209999   95.209999   96.580002   94.360001   95.080002  3438600.0          0.000270  39.772617
#            ZBRA    266.709991  266.709991  268.579987  264.640015  267.670013   700800.0          0.000104  52.322959
#            ZTS     159.270004  159.270004  162.440002  159.080002  161.410004  4243100.0          0.000150  52.741649


df.xs('AAPL', level=1)['rsi'].plot() # to verify that the 'rsi' is indeed volatile as expected
# plt.show() # refer to 'Apple_RSI_Plot.png'


# continue from: 
# https://www.youtube.com/watch?v=9Y3yaoi9rUQ&t=1490s



# level=1 is the 'ticker'
# TBD need to understand what's going on here (consult perplexity)
df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])
df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,1])
df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,2])



# tutor says it requires 2 columns
pandas_ta.atr(high=df['high'], low=df['low'], close=df['close'], length=14)



# TBD what is it about?
# date        ticker
# 2023-05-11  A                NaN
#             AAPL             NaN
#             ABBV             NaN
#             ABNB             NaN
#             ABT              NaN
#                          ...
# 2025-05-09  XYL       112.794805
#             YUM       106.603748
#             ZBH       102.758481
#             ZBRA      107.802160
#             ZTS       107.789862




def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'], low=stock_data['low'], close=stock_data['close'], length=14)
    return atr.sub(atr.mean()).div(atr.std()) # substract the atr mean divides by the standard deviation



# to calculate the ATR index normalized for each stock
df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr) # w/o the group_keys=False we'll gate the 'date' column twice (due to .apply)


# normalizing the data so we can use it for the machine learning module
def compute_macd(close):
    macd = pandas_ta.macd(close=close, length=20).iloc[:,0]
    return macd.sub(macd.mean()).div(macd.std())

df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)  

df['dollar_volume'] = (df['adj close'] * df['volume'])/1e6




##########################################
 # --       continue from URL: https://www.youtube.com/watch?v=9Y3yaoi9rUQ&t=1960s
 # 3r step: to aggregate the data, in a monthloy level,  and filter the top 150 most liquid stocks for each month
 # in order to reduce training time for any potential machine learning model and experiment our features and strategies.
##########################################









# df.unstack('ticker')['dollar_volume'] # this gives the 'dollar_volume' per day (in index) per stock (in columns) - similar to PivotTables
# df.columns
# MultiIndex([('Adj Close',    'A'),
#             ('Adj Close', 'AAPL'),
#             ('Adj Close', 'ABBV'),
#             ('Adj Close', 'ABNB'),
#             ('Adj Close',  'ABT'),
#             ('Adj Close', 'ACGL'),
#             ('Adj Close',  'ACN'),
#             ('Adj Close', 'ADBE'),
#             ('Adj Close',  'ADI'),
#             ('Adj Close',  'ADM'),
#             ...
#             (   'Volume',  'WTW'),
#             (   'Volume',   'WY'),
#             (   'Volume', 'WYNN'),
#             (   'Volume',  'XEL'),
#             (   'Volume',  'XOM'),
#             (   'Volume',  'XYL'),
#             (   'Volume',  'YUM'),
#             (   'Volume',  'ZBH'),
#             (   'Volume', 'ZBRA'),
#             (   'Volume',  'ZTS')],
#            names=['Price', 'Ticker'], length=3018)


# df.iloc[0:5, 0:10]  # df post -> unstack('ticker')['dollar_volume']
# Price        Adj Close                                                                                                          
# Ticker               A        AAPL        ABBV        ABNB         ABT       ACGL         ACN        ADBE         ADI        ADM
# Date
# 2023-05-11  125.803558  171.789978  135.933609  111.199997  105.743347  73.513962  263.458984  341.579987  174.311722  70.256615
# 2023-05-12  125.636024  170.859283  136.452927  105.275002  106.166130  73.124100  268.219757  335.450012  174.292419  70.706848
# 2023-05-15  126.128754  170.364258  135.933609  105.779999  105.541557  71.412476  268.529449  345.670013  177.960312  71.344704
# 2023-05-16  124.453484  170.364258  132.873535  105.410004  105.109169  70.860954  270.155029  345.109985  176.435226  69.023483
# 2023-05-17  125.911964  170.978119  132.929184  108.330002  104.561485  69.453629  275.418976  356.630005  181.425507  68.929115


# df.iloc[0:5, -10:]  
# Price       Volume                                                                                 
# Ticker         WTW       WY     WYNN      XEL       XOM      XYL      YUM      ZBH    ZBRA      ZTS
# Date
# 2023-05-11  254900  3331900  4509700  2850100  17165900  1278700  1144100   899700  349400  1539100
# 2023-05-12  446900  2910300  2653500  2295800  12608300  1242700  1114700   840700  221500  1267700
# 2023-05-15  248000  2415600  2429200  1708600  13715600  1469100   896200   905300  235700  1060600
# 2023-05-16  323600  4424100  2382700  1770200  14795200  1603300  1572200   929400  183700  1175300
# 2023-05-17  413700  4708200  5680500  2118000  14064700  1957400  1690400  1071500  227400  1714500



# df.unstack('ticker')['dollar_volume'] # this gives the 'dollar_volume' per day (in index) per stock (in columns) - similar to PivotTables (see above)
# df.unstack('ticker')['dollar_volume'].resample('M').mean() # resample it to monthly and get the mean - average dollar volume mean per month (month as index)



last_cols = [c for c in df.columns.unique(0) if c not in ['dollar_volume', 'volume', 'open', 'high', 'low', 'close']]



# df.unstack('ticker')['dollar_volume'].resample('M').stack('ticker')# and stack it back into a muti-index and to frame only the 'dollar_volume' columms (TBD what happens if I get only that column? like df[['dollar_volume']])?
data = (pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'), # and stack it back into a muti-index and to frame only the 'dollar_volume' columms (TBD what happens if I get only that column? like df[['dollar_volume']])?
           df.unstack()[last_cols].resample('M').last().stack('ticker')],
          axis=1)).dropna()



data.head()
#                    dollar_volume   adj close  garman-klass_vol        rsi    bb_low    bb_mid   bb_high       atr      macd
# date       ticker
# 2024-06-30 A          388.902169  128.672516          0.000429  33.080377  4.861591  4.885451  4.909310 -0.879331 -1.238910
#            AAPL     18756.901227  209.639572          0.000045  63.396259  5.241509  5.325092  5.408676 -0.296120  1.462088
#            ABBV      1010.346462  165.507538         -0.000144  60.393036  5.053421  5.095719  5.138017 -0.914867  0.488216
#            ABNB       570.367028  151.630005          0.000110  59.104537  4.978593  5.004430  5.030266 -1.258109  0.707400
#            ABT        625.330361  101.867867         -0.000168  47.775896  4.612815  4.640067  4.667320 -0.994178 -0.409451

# no to calculate the NUM_OF_YEARS years rolling average dollar volume per stock and use it to filter out the top 150 most liquid stocks for each month
data['dollar_volume'] = (data.loc[:, 'dollar_volume'].unstack('ticker').rolling(NUM_OF_YEARS*12, min_periods=12).mean().stack())

data

#                    dollar_volume   adj close  garman-klass_vol        rsi    bb_low    bb_mid   bb_high       atr      macd
# date       ticker
# 2023-06-30 A                 NaN  118.727608          0.000200  49.044045  4.739386  4.766929  4.794472 -1.420418 -0.452425
#            AAPL              NaN  192.047180          0.000138  77.002553  5.170929  5.212888  5.254847 -0.902301  1.099188
#            ABBV              NaN  124.935814         -0.001415  40.005252  4.820349  4.846565  4.872782 -0.749221 -0.922895
#            ABNB              NaN  128.160004          0.000490  63.096741  4.748322  4.823852  4.899382  0.439245  1.520310
#            ABT               NaN  104.753662         -0.000246  59.712139  4.573052  4.624646  4.676239 -0.745914  0.125747
# ...                          ...         ...               ...        ...       ...       ...       ...       ...       ...
# 2025-05-31 XYL        163.895740  122.849998          0.000050  57.264323  4.668716  4.762332  4.855948  1.500332  0.811859
#            YUM        252.175143  147.130005          0.000118  48.007863  4.961295  4.994114  5.026934  2.195996 -0.436357
#            ZBH        190.366943   95.209999          0.000270  39.772617  4.524654  4.600213  4.675772  2.546141 -1.244519
#            ZBRA       128.704504  266.709991          0.000104  52.322959  5.381005  5.494977  5.608949  1.218764  0.046676
#            ZTS        422.714762  159.270004          0.000150  52.741649  4.980900  5.036342  5.091783  2.064624  0.256402


data['dollar_vol_rank'] = data.groupby('date')['dollar_volume'].rank(ascending=False) # high rank# to small volume and vise-versa (thanks to the ascending=False)
data[data['dollar_vol_rank']<150] # selecting the top 150 liquid stock per month (#1 - #5  are actually the highest ranked stocks and we want the high 150 stocks, axis=1), axis=1

# and removing the utility columns

data = data[data['dollar_vol_rank']<150].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)

# calculating monthly returns for different time horizons and add them as additional features to what we already have
# capturing the momentum patterns

# g = df.xs('AAPL', level=1)

def calculate_returns(df):

    outlier_cutoff = 0.005

    lags = [1, 2, 3, 6, 9, 12] # patterns monthly, quarterly, half year, 9 months and 1 year

    for lag in lags:
        df[f'return_{lag}m'] = (df['adj close']
                              .pct_change(lag) # percentile change
                              .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff), upper=x.quantile(1-outlier_cutoff)))
                              .add(1)
                              .pow(1/lag)
                              .sub(1)) # note that without the parenthesis it wouldn't work!
    return df



# def calculate_returns(df):

#     outlier_cutoff = 0.005

#     lags = [1, 2, 3, 6, 9, 12] # patterns monthly, quarterly, half year, 9 months and 1 year

#     for lag in lags:
#         df[f'return_{lag}m'] = (df['adj close'].pct_change(lag).pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff), upper=x.quantile(1-outlier_cutoff))).add(1).pow(1/lag).sub(1))
#     return df



# data = data.groupby('ticker', group_keys=False).apply(calculate_returns).dropna()
data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()


# gives a dict() with keys [0,1] each has DataFrame
web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2010')

web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2010')[0].info()
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   Mkt-RF  180 non-null    float64
#  1   SMB     180 non-null    float64
#  2   HML     180 non-null    float64
#  3   RMW     180 non-null    float64
#  4   CMA     180 non-null    float64
#  5   RF      180 non-null    float64


# each month is an index
web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2010')[0].index
# PeriodIndex(['2010-01', '2010-02', '2010-03', '2010-04', '2010-05', '2010-06',
#              '2010-07', '2010-08', '2010-09', '2010-10',
#              ...
#              '2024-03', '2024-04', '2024-05', '2024-06', '2024-07', '2024-08',
#              '2024-09', '2024-10', '2024-11', '2024-12'],
#             dtype='period[M]', name='Date', length=180)



# and dropping the 'RF' factor
factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2012')[0].drop('RF', axis=1)


# factor_data.index = pd.to_datetime(factor_data.index) # this didn't work, so we tried the below option
factor_data.index = factor_data.index.to_timestamp() # to align
# DatetimeIndex(['2010-01-01', '2010-02-01', '2010-03-01', '2010-04-01',
#                '2010-05-01', '2010-06-01', '2010-07-01', '2010-08-01',
#                '2010-09-01', '2010-10-01',
#                ...
#                '2024-03-01', '2024-04-01', '2024-05-01', '2024-06-01',
#                '2024-07-01', '2024-08-01', '2024-09-01', '2024-10-01',
#                '2024-11-01', '2024-12-01'],
#               dtype='datetime64[ns]', name='Date', length=180, freq='MS')


# getting the last value per month instead of the first by (followed by changing to percentile value):
factor_data = factor_data.resample('M').last().div(100)
#             Mkt-RF     SMB     HML     RMW     CMA      RF
# Date
# 2010-01-31 -0.0336  0.0034  0.0043 -0.0127  0.0046  0.0000
# 2010-02-28  0.0340  0.0151  0.0322 -0.0027  0.0142  0.0000
# 2010-03-31  0.0631  0.0185  0.0221 -0.0065  0.0169  0.0001
# 2010-04-30  0.0200  0.0498  0.0289  0.0069  0.0172  0.0001
# 2010-05-31 -0.0789  0.0005 -0.0244  0.0130 -0.0022  0.0001
# ...            ...     ...     ...     ...     ...     ...
# 2024-08-31  0.0161 -0.0365 -0.0113  0.0085  0.0086  0.0048
# 2024-09-30  0.0174 -0.0102 -0.0259  0.0004 -0.0026  0.0040
# 2024-10-31 -0.0097 -0.0088  0.0089 -0.0138  0.0103  0.0039
# 2024-11-30  0.0651  0.0478 -0.0005 -0.0262 -0.0217  0.0040
# 2024-12-31 -0.0317 -0.0387 -0.0295  0.0182 -0.0110  0.0037


factor_data.index.name = 'date' #instead of .lower()  - for consistency

factor_data = factor_data.join(data['return_1m']).sort_index()

factor_data                                                                                                                                             
#                    Mkt-RF     SMB     HML     RMW     CMA  return_1m
# date       ticker
# 2022-05-31 AAPL   -0.0034 -0.0016  0.0859  0.0170  0.0399  -0.054496
#            ABBV   -0.0034 -0.0016  0.0859  0.0170  0.0399   0.003336
#            ABT    -0.0034 -0.0016  0.0859  0.0170  0.0399   0.034890
#            ACN    -0.0034 -0.0016  0.0859  0.0170  0.0399  -0.006326
#            ADBE   -0.0034 -0.0016  0.0859  0.0170  0.0399   0.051850
# ...                   ...     ...     ...     ...     ...        ...
# 2024-12-31 VZ     -0.0317 -0.0387 -0.0295  0.0182 -0.0110  -0.097335
#            WDAY   -0.0317 -0.0387 -0.0295  0.0182 -0.0110   0.032161
#            WFC    -0.0317 -0.0387 -0.0295  0.0182 -0.0110  -0.077852
#            WMT    -0.0317 -0.0387 -0.0295  0.0182 -0.0110  -0.021079
#            XOM    -0.0317 -0.0387 -0.0295  0.0182 -0.0110  -0.088081



factor_data.xs('AAPL', level='ticker').head()                                                                            
#             Mkt-RF     SMB     HML     RMW     CMA  return_1m
# date
# 2022-05-31 -0.0034 -0.0016  0.0859  0.0170  0.0399  -0.054496
# 2022-06-30 -0.0844  0.0136 -0.0610  0.0174 -0.0472  -0.081430
# 2022-07-31  0.0957  0.0183 -0.0403  0.0085 -0.0682   0.174907
# 2022-08-31 -0.0377  0.0152  0.0029 -0.0479  0.0133  -0.031208
# 2022-09-30 -0.0935 -0.0104  0.0002 -0.0146 -0.0079  -0.120977




factor_data.xs('MSFT', level='ticker').head() # same factors as in above, different returns (self checkup)
#             Mkt-RF     SMB     HML     RMW     CMA  return_1m                                                                                                 
# date                                                                                                                                                          
# 2022-05-31 -0.0034 -0.0016  0.0859  0.0170  0.0399  -0.018077
# 2022-06-30 -0.0844  0.0136 -0.0610  0.0174 -0.0472  -0.055321
# 2022-07-31  0.0957  0.0183 -0.0403  0.0085 -0.0682   0.093097
# 2022-08-31 -0.0377  0.0152  0.0029 -0.0479  0.0133  -0.066663
# 2022-09-30 -0.0935 -0.0104  0.0002 -0.0146 -0.0079  -0.107058




factor_data.xs('ALGN', level='ticker') # find below that ALGN has only 7 months of data total
#             Mkt-RF     SMB     HML     RMW     CMA  return_1m
# date
# 2022-05-31 -0.0034 -0.0016  0.0859  0.0170  0.0399  -0.042323
# 2022-06-30 -0.0844  0.0136 -0.0610  0.0174 -0.0472  -0.147565
# 2022-07-31  0.0957  0.0183 -0.0403  0.0085 -0.0682   0.183067
# 2022-08-31 -0.0377  0.0152  0.0029 -0.0479  0.0133  -0.132648
# 2022-09-30 -0.0935 -0.0104  0.0002 -0.0146 -0.0079  -0.150144
# 2022-10-31  0.0783  0.0188  0.0806  0.0331  0.0662  -0.061851
# 2022-11-30  0.0461 -0.0275  0.0141  0.0632  0.0320   0.012146



# filter out stocks with less than 10 months of data (we get a 'Series')
observations = factor_data.groupby(level='ticker').size() #  we get max 32 months and he gets 71 months
# ticker                                                                                                                                                        
# AAPL    32                                                                                                                                                    
# ABBV    32                                                                                                                                                    
# ABNB    25                                                                                                                                                    
# ABT     32                                                                                                                                                    
# ACN     32                                                                                                                                                    
#         ..                                                                                                                                                    
# WDAY    32                                                                                                                                                    
# WFC     32                                                                                                                                                    
# WMT     32                                                                                                                                                    
# WYNN     2                                                                                                                                                    
# XOM     32  

valid_stocks = observations[observations >= 10]
# valid_stocks = observations[observations <= 10]
valid_stocks.index.to_list() # 150 stocks
# 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACN', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AMAT', 'AMD', 'AMGN', 'AMT', 'AMZN', 'AON', 'AVGO', 'AXP', 'BA', 'BAC', 'BIIB', 'BKNG', 'BLK', 'BMY', 'BRK-B', 'BSX', 'BX', 'C', 'CAT', 'CCL', 'CHTR', 'CI', 'CMCSA', 'CMG', 'COF', 'COP', 'COST', 'CRM', 'CRWD', 'CSCO', 'CSX', 'CVS', 'CVX', 'DAL', 'DASH', 'DE', 'DG', 'DHR', 'DIS', 'DVN', 'DXCM', 'EBAY', 'EL', 'ELV', 'ENPH', 'EQIX', 'EXPE', 'F', 'FCX', 'FDX', 'FI', 'FIS', 'GE', 'GILD', 'GM', 'GOOG', 'GOOGL', 'GS', 'HD', 'HON', 'HUM', 'IBM', 'INTC', 'INTU', 'ISRG', 'JNJ', 'JPM', 'KLAC', 'KO', 'LIN', 'LLY', 'LMT', 'LOW', 'LRCX', 'LULU', 'LUV', 'MA', 'MAR', 'MCD', 'MDLZ', 'MDT', 'META', 'MMM', 'MPC', 'MRK', 'MRNA', 'MS', 'MSFT', 'MTCH', 'MU', 'NCLH', 'NEE', 'NEM', 'NFLX', 'NKE', 'NOC', 'NOW', 'NVDA', 'NXPI', 'ON', 'ORCL', 'OXY', 'PANW', 'PARA', 'PEP', 'PFE', 'PG', 'PLD', 'PLTR', 'PM', 'PYPL', 'QCOM', 'RCL', 'REGN', 'RTX', 'SBUX', 'SCHW', 'SHW', 'SLB', 'SPGI', 'T', 'TGT', 'TJX', 'TMO', 'TMUS', 'TSLA', 'TXN', 'UAL', 'UBER', 'UNH', 'UNP', 'UPS', 'V', 'VLO', 'VRTX', 'VZ', 'WDAY', 'WFC', 'WMT', 'XOM'

factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]




# Calculating Rolling Factor Betas

betas = (factor_data.groupby(level='ticker', group_keys=False)
                    .apply(lambda x: RollingOLS(endog=x['return_1m'], 
                                                exog=sm.add_constant(x.drop('return_1m', axis=1)), # we used the 'return_1m' and then dropped it
                                                window=min(24, x.shape[0]), # windows of 24 or min size (if reminder)
                                                min_nobs=len(x.columns)+1)
                    .fit(params_only=True)
                    .params
                    .drop('const', axis=1)))



betas
#                      Mkt-RF       SMB       HML       RMW       CMA
# date       ticker
# 2022-05-31 AAPL         NaN       NaN       NaN       NaN       NaN
#            ABBV         NaN       NaN       NaN       NaN       NaN
#            ABT          NaN       NaN       NaN       NaN       NaN
#            ACN          NaN       NaN       NaN       NaN       NaN
#            ADBE         NaN       NaN       NaN       NaN       NaN
# ...                     ...       ...       ...       ...       ...
# 2024-12-31 VZ      1.224522 -0.886299  0.813497  0.661615 -0.506159
#            WDAY    0.919440 -1.203245  0.474934 -2.139354 -1.488875
#            WFC     0.438125  0.046503  0.802104 -2.210930 -0.014847
#            WMT     0.740271  0.331429 -0.559902  1.171367  0.734943
#            XOM     0.685566  0.042542  0.254305  1.153105  0.714075




data = (data.join(betas.groupby('ticker').shift()))

FF_Factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

data.loc[:, FF_Factors] = data.groupby('ticker', group_keys=False)[FF_Factors].apply(lambda x: x.fillna(x.mean()))



data.drop('adj close', axis=1, inplace=True)

data.dropna(inplace=True)

data.info()
#  #   Column            Non-Null Count  Dtype
# ---  ------            --------------  -----
#  0   garman-klass_vol  4930 non-null   float64
#  1   rsi               4930 non-null   float64
#  2   bb_low            4930 non-null   float64
#  3   bb_mid            4930 non-null   float64
#  4   bb_high           4930 non-null   float64
#  5   atr               4930 non-null   float64
#  6   macd              4930 non-null   float64
#  7   return_1m         4930 non-null   float64
#  8   return_2m         4930 non-null   float64
#  9   return_3m         4930 non-null   float64
#  10  return_6m         4930 non-null   float64
#  11  return_9m         4930 non-null   float64
#  12  return_12m        4930 non-null   float64
#  13  Mkt-RF            4930 non-null   float64
#  14  SMB               4930 non-null   float64
#  15  HML               4930 non-null   float64
#  16  RMW               4930 non-null   float64
#  17  CMA               4930 non-null   float64


