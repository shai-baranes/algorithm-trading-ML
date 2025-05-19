import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates






np.random.seed(0)
dates = pd.date_range('2023-01-01', periods=90)
items = ['A', 'B']
data = np.random.randint(1, 10, size=(90, 2))  # array of arrays, each internal array comprised of 3 values.
df = pd.DataFrame(data, index=dates, columns=items)

# TBD - replace A/B/C with stocks

df
#             A  B
# 2023-01-01  6  1
# 2023-01-02  4  4
# 2023-01-03  8  4
# 2023-01-04  6  3
# 2023-01-05  5  8
# ...        .. ..
# 2023-03-27  7  6
# 2023-03-28  8  1
# 2023-03-29  9  5
# 2023-03-30  7  6
# 2023-03-31  9  3


df.info()
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   A       90 non-null     int32
#  1   B       90 non-null     int32

df.columns
# Index(['A', 'B'], dtype='object')

df.index
# DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04',
#                '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08',
#                '2023-01-09', '2023-01-10', '2023-01-11', '2023-01-12',
#                '2023-01-13', '2023-01-14', '2023-01-15', '2023-01-16',
#                ...........
#                '2023-03-18', '2023-03-19', '2023-03-20', '2023-03-21',
#                '2023-03-22', '2023-03-23', '2023-03-24', '2023-03-25',
#                '2023-03-26', '2023-03-27', '2023-03-28', '2023-03-29',
#                '2023-03-30', '2023-03-31'],
#               dtype='datetime64[ns]', freq='D')

df.stack().index
# MultiIndex([('2023-01-01', 'A'),    -->  6  (from above randmized values)
#             ('2023-01-01', 'B'),    -->  1
#             ('2023-01-05', 'A'),
#             ('2023-01-05', 'B'),
#             ...
#             ('2023-03-27', 'A'),
#             ('2023-03-31', 'B')],
#            length=180)

df.stack()[('2023-01-01', 'A')] # 6 # note that stack()/unstack() is not 'inplace=True'





# data = (pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'), # and stack it back into a muti-index and to frame only the 'dollar_volume' columms (TBD what happens if I get only that column? like df[['dollar_volume']])?
#            df.unstack()[last_cols].resample('M').last().stack('ticker')],
#           axis=1)).dropna()




# now with additional value (to be able to unstack by a specific 'key')
np.random.seed(0)
dates = pd.date_range('2023-01-01', periods=90)
items = ['Apple', 'Binance', 'CMT']
data = np.random.randint(1, 10, size=(90, 3))  # array of arrays, each internal array comprised of 3 values.
df = pd.DataFrame(data, index=dates, columns=items)

df
#             Apple Binance  CMT
# 2023-01-01  6     1        4
# 2023-01-02  4     8        4
# 2023-01-03  6     3        5
# 2023-01-04  8     7        9
# 2023-01-05  9     2        7
# ...        .. .   . .      .
# 2023-03-27  9     9        8
# 2023-03-28  1     4        9
# 2023-03-29  8     8        2
# 2023-03-30  9     5        8
# 2023-03-31  1     5        1



# TBD try this layter using group()
# note that for unstack() you can also use level, either by 
pd.DataFrame(df.unstack(), columns=['A_value']) # in this case, the ressult is pd.Series
#                         A_value
# Apple  2023-01-01       6
#        2023-01-02       4
#        2023-01-03       6
#        2023-01-04       8
#        2023-01-05       9
# .     ..              ...
# CMT    2023-03-27       8
#        2023-03-28       9
#        2023-03-29       2
#        2023-03-30       8
#        2023-03-31       1

type(pd.DataFrame(df.unstack(), columns=['A_value']))


my_df = pd.DataFrame(df.unstack(), columns=['A_value']) # converting to DF with column name
my_df.index.names = ['Stock', 'date'] # adding names to index
my_df

my_df
#                    A_value
# Stock   date              
# Apple  2023-01-01       6
#        2023-01-02       4
#        2023-01-03       6
#        2023-01-04       8
#        2023-01-05       9
# ...                   ...
# CMT    2023-03-27       8
#        2023-03-28       9
#        2023-03-29       2
#        2023-03-30       8
#        2023-03-31       1


my_df.swaplevel()
#                    A_value
# date       Stock        
# 2023-01-01 Apple        6
# 2023-01-02 Apple        4
# 2023-01-03 Apple        6
# 2023-01-04 Apple        8
# 2023-01-05 Apple        9
# ...                   ...
# 2023-03-27 CMT          8
# 2023-03-28 CMT          9
# 2023-03-29 CMT          2
# 2023-03-30 CMT          8
# 2023-03-31 CMT          1


my_df = my_df.swaplevel().sort_index()
#                    A_value
# date       Stock        
# 2023-01-01 Apple        6
#            Binance      1
#            CMT          4
# 2023-01-02 Apple        4
#            Binance      8
# ...                   ...
# 2023-03-30 Binance      5
#            CMT          8
# 2023-03-31 Apple        1
#            Binance      5
#            CMT          1

my_df.index # INAGINE THAT THE Stock IS THE STOCK TICKER
# MultiIndex([('2023-01-01', 'Apple'),
#             ('2023-01-01', 'Binance'),
#             ('2023-01-01', 'CMT'),
#             ('2023-01-02', 'Apple'),
#             ('2023-01-02', 'Binance'),
#             ('2023-01-02', 'CMT'),
#             ('2023-01-03', 'Apple'),
#              .........


data = np.random.randint(100, 200, size=(my_df.shape[0], 2))  # array of arrays, each internal array comprised of 3 values.

my_df[['B_value', 'C_value']] = data


my_df
#                    A_value  B_value  C_value
# date       Stock                          
# 2023-01-01 Apple        6      142      114
#            Binance      1      186      128
#            CMT          4      120      182
# 2023-01-02 Apple        4      168      122
#            Binance      8      199      183
# ...                   ...      ...      ...
# 2023-03-30 Binance      5      184      156
#            CMT          8      162      156
# 2023-03-31 Apple        1      148      117
#            Binance      5      157      109
#            CMT          1      162      179

my_df.loc[('2023-01-01', 'Apple'), :]
# values       6
# B_value    142
# C_value    114




# note that when using unstack, you apply per index name or level if not named

# both belowo equivalents
my_df.unstack(level=0)
my_df.unstack(level='date')
#            values                        ...    C_value                      
# date   2023-01-01 2023-01-02 2023-01-03  ... 2023-03-29 2023-03-30 2023-03-31
# Stock                                   ...                                 
# Apple           6          4          6  ...        179        173        117
# Binance         1          8          3  ...        103        156        109
# CMT             4          4          5  ...        167        156        179



# note that resample works only on a datetime index type
# here we get the montly average (end of each month) on a period of 3 months
my_df.unstack().resample('ME').mean()
#               values                      ...     C_value                        
# Stock        Apple    Binance    CMT      ...   Apple       Binance     CMT
# date                                      ...                                    
# 2023-01-31  5.193548  4.548387  4.483871  ...  147.806452  145.225806  154.387097
# 2023-02-28  4.464286  4.964286  6.178571  ...  150.500000  157.642857  156.535714
# 2023-03-31  4.419355  5.225806  5.129032  ...  138.322581  157.451613  144.258065



# here we get the max values on a period of 3 months
my_df.unstack().resample('ME').max()
#                   A_values                    B_value                          C_value          
# Stock          Apple  Binance  CMT       Apple    Binance    CMT         Apple    Binance    CMT
# date                                                       
# 2023-01-31      9      9        9         198        199     192          199       197      194
# 2023-02-28      9      9        9         194        198     193          198       197      198
# 2023-03-31      9      9        9         189        194     196          190       199      199



# here we get the sum of the values over a period of 3 months
my_df.unstack().resample('ME').sum()
#                   A_value           B_value             C_value            
# Stock          Ap Bin   CMT     Ap   Bin   CMT      Ap  Bin     CMT
# date                                                               
# 2023-01-31    161  141  139    4738  4566  4569    4582  4502  4786
# 2023-02-28    125  139  173    4231  4289  4241    4214  4414  4383
# 2023-03-31    137  162  159    4555  4419  4650    4288  4881  4472



# pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume')

my_df.unstack().resample('ME').sum().stack()
#                    A_value  B_value  C_value
# date       Stock                          
# 2023-01-31 Apple      161     4738     4582
#            Binance    141     4566     4502
#            CMT        139     4569     4786
# 2023-02-28 Apple      125     4231     4214
#            Binance    139     4289     4414
#            CMT        173     4241     4383
# 2023-03-31 Apple      137     4555     4288
#            Binance    162     4419     4881
#            CMT        159     4650     4472


my_df.unstack().resample('ME').mean().stack()
#                        A_value     B_value     C_value
# date       Stock                                  
# 2023-01-31 Apple       5.193548  152.838710  147.806452
#            Binance     4.548387  147.290323  145.225806
#            CMT         4.483871  147.387097  154.387097
# 2023-02-28 Apple       4.464286  151.107143  150.500000
#            Binance     4.964286  153.178571  157.642857
#            CMT         6.178571  151.464286  156.535714
# 2023-03-31 Apple       4.419355  146.935484  138.322581
#            Binance     5.225806  142.548387  157.451613
#            CMT         5.129032  150.000000  144.258065

my_df.unstack()['A_value'].resample('ME').mean().stack() # pandas.core.series.Series
# date        Stock
# 2023-01-31  Apple         5.193548
#             Binance       4.548387
#             CMT           4.483871
# 2023-02-28  Apple         4.464286
#             Binance       4.964286
#             CMT           6.178571
# 2023-03-31  Apple         4.419355
#             Binance       5.225806
#             CMT           5.129032


# if I want DF from above with a column name, I frame it as following:
my_df.unstack()['A_value'].resample('ME').mean().stack().to_frame('A_value') #..columns -> Index(['A_value'], dtype='object'), type pandas df
#                       A_value
# date       Stock          
# 2023-01-31 Apple       5.193548
#            Binance     4.548387
#            CMT         4.483871
# 2023-02-28 Apple       4.464286
#            Binance     4.964286
#            CMT         6.178571
# 2023-03-31 Apple       4.419355
#            Binance     5.225806
#            CMT         5.129032



my_df['A_value_percentile'] = my_df.groupby(level='Stock').transform(lambda x: 100*(x/x.sum())).loc[:,'A_value']
# my_df.drop(['B_value', 'C_value'], axis=1).groupby(level='Stock').transform(lambda x: 100*(x/x.sum()))  # another option to view only what I want without changing it
#                       A_value  B_value  C_value     A_value_percentile  # per Stock group
# date       letter                                                
# 2023-01-01 Apple          6      142      114            1.418440
#            Binance        1      186      128            0.226244
#            CMA            4      120      182            0.849257
# 2023-01-02 Apple          4      168      122            0.945626
#            Binance        8      199      183            1.809955
# ...                     ...      ...      ...                 ...
# 2023-03-30 Binance        5      184      156            1.131222
#            CMA            8      162      156            1.698514
# 2023-03-31 Apple          1      148      117            0.236407
#            Binance        5      157      109            1.131222
#            CMA            1      162      179            0.212314





formatted_ticks = [item.strftime('%Y-%m-%d') for item in my_df.unstack().resample('ME').mean().index.to_list()]
# ['2023-01-31', '2023-02-28', '2023-03-31']

ax = my_df.drop('A_value_percentile', axis=1).unstack().resample('ME').mean().plot(kind='bar', subplots=True, rot=60, figsize=(12, 9), layout=(3, 3))
# ax = my_df.unstack().resample('ME').mean().plot(kind='bar', subplots=True, rot=60, figsize=(12, 9), layout=(4, 3))
plt.xticks(ticks=range(len(formatted_ticks)), labels=formatted_ticks) # my way to remove the 00:00:00 tail





plt.tight_layout()

plt.show() # Stock_M_Resample_Figure_1.png



# df['percentile'] = df.groupby('ticker')['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))
my_df.groupby(level='Stock', group_keys=False)['A_value'].apply(lambda x: 100*(x/x.sum())) # we get double 'letter' (group_keys=False to avoid that)
#                            A_value   B_value   C_value
# letter date       letter                              
# Apple  2023-01-01 Apple   1.418440  1.049985  0.871293
#        2023-01-02 Apple   0.945626  1.242236  0.932437
#        2023-01-03 Apple   1.418440  1.190476  0.863650
#        2023-01-04 Apple   1.891253  1.323573  1.169367
#        2023-01-05 Apple   2.127660  1.072168  0.963008
# ...                            ...       ...       ...
# CMT    2023-03-27 CMT     1.698514  0.936107  1.341544
#        2023-03-28 CMT     1.910828  0.958395  0.733084
#        2023-03-29 CMT     0.424628  1.196137  1.224250
#        2023-03-30 CMT     1.698514  1.203566  1.143611
#        2023-03-31 CMT     0.212314  1.203566  1.312221

#equivalent to above without needing the: (group_keys=False)
my_df.groupby(level='Stock')['A_value'].transform(lambda x: 100*(x/x.sum())).tail


# date       letter                               
# 2023-01-01 Apple    1.418440  1.049985  0.871293
#            Binance  0.226244  1.401235  0.927738
#            CMT      0.849257  0.891530  1.334213
# 2023-01-02 Apple    0.945626  1.242236  0.932437
#            Binance  1.809955  1.499171  1.326375
# ...                      ...       ...       ...
# 2023-03-30 Binance  1.131222  1.386168  1.130681
#            CMT      1.698514  1.203566  1.143611
# 2023-03-31 Apple    0.236407  1.094351  0.894222
#            Binance  1.131222  1.182763  0.790027
#            CMT      0.212314  1.203566  1.312221


import pdb; pdb.set_trace()  # breakpoint 9e09c1f6 //
