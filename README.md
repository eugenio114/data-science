# This repository contains my data science projects 

- **detrending-time-series** : this file shows how to detrend a time series dataset by model fitting (in this case I have used some dummy data of EMEA sales volumes) using scikit-learn Linear Regression model. 
- **forecasting-sales-volumes-sarimax** : this file contains a predictive SARIMAX model that forecasts sales volume of EMEA, AMERICA and ASIA for 6 month time horizon. 
  The SARIMAX model is totaly based on historical patterns of actual sales quantity ('monthly-sales-data-final.csv').
  This model best performs under the assumption that "the future" will follow very similar pattern of what already experienced in the past and that there will not be any major event causing sales figures to considerably deviate from this pattern.
