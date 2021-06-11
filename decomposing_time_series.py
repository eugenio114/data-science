import pandas as pd
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

#####READS THE MONTHLY SALES DATA CSV FILE
sales = pd.read_csv('./data_csv/monthly-sales-data-final.csv')

#####SET MONTH AS INDEX FOR DECOMPOSITION AND FILTER FOR ONLY NORTH AMERICA SALES DATA
america_sales_actuals = sales.loc[sales['region'] == 'AMERICA'].set_index('month')


#####PLOT THE TIME SERIES OF SALES DATA
america_sales_actuals.plot(figsize=(16, 8), title = 'AMERICA Sales Actuals by month')
plt.xticks(rotation=90)
plt.show()

import statsmodels.api as sm
from pylab import rcParams

rcParams['figure.figsize'] = 16, 8

#####STORE SALES VALUES INTO NEW SERIES FOR DECOMPOSITION
america_sales = (america_sales_actuals['sales_quantity']).reset_index().drop(columns=['month'])

AMERICA_decomposition = sm.tsa.seasonal_decompose(america_sales, model='multiplicative', period=12, extrapolate_trend='freq')
fig = AMERICA_decomposition.plot()
plt.show()

