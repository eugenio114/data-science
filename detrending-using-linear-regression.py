import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

emea_sales = pd.read_csv('./data_csv/emea-monthly-sales-data.csv')

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['text.color'] = 'k'


###PLOTTING THE TIME SERIES###
emea_sales.plot(x='month', y='sales_quantity', label='EMEA SALES', figsize=(16,6))
plt.xticks(rotation=90)
plt.show()

#### TO KEEP EMEA SALES VALUES ONLY
emea_sales_values = emea_sales['sales_quantity']

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
# detrend
##########MODiFY NAME BELOW
emea_detrended_sales = [y[i]-trend[i] for i in range(0, len(emea_sales_series))] #CHANGE HERE THE TIME SERIES_________
# plot detrended
plt.plot(emea_detrended_sales) #CHANGE HERE THE TIME SERIES_________
plt.show()
