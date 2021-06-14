import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

emea_sales = pd.read_csv('./data_csv/emea-monthly-sales-data.csv')
emea_orders = pd.read_csv('./data_csv/emea-monthly-orders-data.csv')

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['text.color'] = 'k'


###PLOTTING THE TIME SERIES OF SALES QUANTITY AND ORDERS NUMBER###
emea_sales.plot(x='month', y='sales_quantity', label='EMEA SALES', figsize=(16,6))
plt.xticks(rotation=90)
plt.show()

emea_orders.plot(x='month', y='orders_num', label='EMEA ORDERS', figsize=(16,6))
plt.xticks(rotation=90)
plt.show()

#### TO KEEP EMEA SALES AND THEN EMEA VALUES ONLY
emea_sales_values = emea_sales['sales_quantity']
emea_orders_values = emea_orders['orders_num']

#### DETRENDING THE TIME SERIES OF EMEA SALES USING MODEL FITTING
from sklearn.linear_model import LinearRegression
#numpy.squeeze() function is used when we want to remove single-dimensional entries from the shape of an array.
emea_sales_series = emea_sales_values.squeeze()

# fit linear model
X = [i for i in range(0, len(emea_sales_series))]  #CHANGE HERE THE TIME SERIES_________
X = np.reshape(X, (len(X), 1))
y = emea_sales_series.values #CHANGE HERE THE TIME SERIES_________
model = LinearRegression()
model.fit(X, y)
# calculate trend
trend = model.predict(X)
# plot trend
plt.plot(y)
plt.plot(trend)
plt.show()
#detrend
##########MODiFY NAME BELOW
emea_detrended_sales = [y[i]-trend[i] for i in range(0, len(emea_sales_series))] #CHANGE HERE THE TIME SERIES_________
# plot detrended sales
plt.plot(emea_detrended_sales) #CHANGE HERE THE TIME SERIES_________
plt.show()

#### DETRENDING THE TIME SERIES OF EMEA ORDERS USING MODEL FITTING
#numpy.squeeze() function is used when we want to remove single-dimensional entries from the shape of an array.
emea_orders_series = emea_orders_values.squeeze()

# fit linear model
X = [i for i in range(0, len(emea_orders_series))]  #CHANGE HERE THE TIME SERIES_________
X = np.reshape(X, (len(X), 1))
y = emea_orders_series.values #CHANGE HERE THE TIME SERIES_________
model = LinearRegression()
model.fit(X, y)
# calculate trend
trend = model.predict(X)
# plot trend
plt.plot(y)
plt.plot(trend)
plt.show()
#detrend
##########MODiFY NAME BELOW
emea_detrended_orders = [y[i]-trend[i] for i in range(0, len(emea_orders_series))] #CHANGE HERE THE TIME SERIES_________
# plot detrended sales
plt.plot(emea_detrended_orders) #CHANGE HERE THE TIME SERIES_________
plt.show()


#IMPORT TO BE ABLE TO RUN THE CROSS CORRELATION FUNCTION (CCF)
from statsmodels.tsa.stattools import ccf

ccf_starts_oa_emea = plt.plot(ccf(emea_detrended_sales, emea_detrended_orders, unbiased=True)) #CHANGE HERE THE TIME SERIES______

#DEFINING THE VARIABLES FOR THE XCORR
x = pd.Series(emea_detrended_sales).astype('float') #CHANGE HERE THE TIME SERIES_________

y = pd.Series(emea_detrended_orders).astype('float') #CHANGE HERE THE TIME SERIES_________

fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
emea_sales_orders = ax1.xcorr(x.astype('float'), y.astype('float'), usevlines=True, maxlags=10, normed=True, lw=2) #CHANGE HERE THE TIME SERIES_________

ax1.grid(True)

ax2.acorr(x, usevlines=True, normed=True, maxlags=10, lw=2)
ax2.grid(True)

plt.show()

#STORING THE CORRELATION VALUES IN A DEDICATED DATAFRAME
emea_df_sales_orders_corr = pd.DataFrame(pd.Series(emea_sales_orders[0], emea_sales_orders[1])).reset_index().rename(columns={'index': 'emea_apps_corr', 0: 'period'})
#TO PRINT THE CORRELATION VALUES PER PERIOD
print(emea_df_sales_orders_corr)