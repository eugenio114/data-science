import pandas as pd
import numpy as np
import os
import sklearn
import warnings
from sklearn.metrics import accuracy_score
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

#####READS THE MONTHLY SALES DATA CSV FILE
sales = pd.read_csv('./data_csv/monthly-sales-data.csv')

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

