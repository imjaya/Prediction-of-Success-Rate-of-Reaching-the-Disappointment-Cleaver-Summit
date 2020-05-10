import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor
import statsmodels.api as sm
from pylab import rcParams
import warnings
import itertools
from sklearn.linear_model import LinearRegression


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans



data_climb = pd.read_csv('D:/SPRING 2020/EEE 511 Artificial Neural Computation/Projects/contest/climbing_statistics.csv')
data_weather = pd.read_csv('D:/SPRING 2020/EEE 511 Artificial Neural Computation/Projects/contest/Rainier_Weather.csv')
data_climb.head()
data_weather.head()
data_climb.info()
data_weather.info()

data_climb.Date.drop_duplicates().count()
data_weather['Date'] = pd.to_datetime(data_weather['Date'])
data_weather.info()

data_weather.head()
data_weather['Date'].max()
data_climb['Date'] = pd.to_datetime(data_climb['Date'])
data_climb.info()
data_climb.head()



s=data_climb.groupby(['Date'])[ 'Attempted' , 'Succeeded' , 'Success Percentage'].agg({'Attempted' :'sum' ,'Succeeded' :'sum' ,'Success Percentage':'mean' , 'Route' : lambda x:x.value_counts().index[0]}).sort_values(by = 'Attempted' , ascending=False)
print(s)
s['Success Percentage'] = s['Succeeded'] / s['Attempted']
s.info()
s_pooled = pd.merge(right=data_weather  , left = s , how = 'outer', on ='Date')
s_pooled.info()
s_pooled.head()


for i in ['Attempted' , 'Succeeded' , 'Success Percentage'    ] :

    s_pooled[i] = np.where ((np.isnan(s_pooled[i]) == 1) &  (np.isnan(s_pooled['Battery Voltage AVG']) == 0) , 0 , s_pooled[i])


s_pooled.info()


s_pooled=s_pooled.set_index('Date')
Attempt = s_pooled['Attempted']
Attempt.head()

Attempt.plot(figsize=(15, 6))
plt.show()


Attempt.index


decomposition = sm.tsa.seasonal_decompose(Attempt, model='additive' , freq = 30)
decomposition.plot()

plt.show()



p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x{}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


import warnings
warnings.simplefilter(action='ignore')
a =[]
b=[]
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(Attempt,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            c=param
            e=param_seasonal
            d=results.aic
            a.append((c,e))
            b.append(d)
        except:
            continue


models=pd.DataFrame({'model':a,'AIC':b})
models.head()
models.loc[models['AIC'] == models.AIC.min(),:]


mod = sm.tsa.statespace.SARIMAX(Attempt, order=(1, 0, 1), seasonal_order=(0, 0,1, 12), enforce_stationarity=False,)
results = mod.fit()
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(16, 8))
plt.show()

features =['Battery Voltage AVG' , 'Temperature AVG' ,'Relative Humidity AVG' ,'Wind Speed Daily AVG' ,'Wind Direction AVG','Solare Radiation AVG']
for i in features:
  print('distribution of ' , i)
  print(s_pooled.loc[(np.isnan(s_pooled[i]) ==0)][i].describe())
  sns.distplot(s_pooled.loc[(np.isnan(s_pooled[i]) ==0)][i])
  plt.show()

s_pooled2=s_pooled.loc[ :, ['Battery Voltage AVG' , 'Temperature AVG' ,'Relative Humidity AVG' ,'Wind Speed Daily AVG' ,'Wind Direction AVG','Solare Radiation AVG','Success Percentage' , 'Route']]
s_pooled2.head()
s_pooled2.info()
s_pooled2['Route'].value_counts()
s_pooled2 = pd.get_dummies(s_pooled2 , columns=['Route'])

s_pooled2.loc[np.isnan(s_pooled2['Battery Voltage AVG']) ==0,['Battery Voltage AVG','Temperature AVG','Relative Humidity AVG','Wind Speed Daily AVG','Wind Direction AVG'
                ,'Solare Radiation AVG','Success Percentage']].corr()


s_pooled2.head()

x_train = s_pooled2.loc[np.isnan(s_pooled2['Battery Voltage AVG']) ==0,['Battery Voltage AVG','Relative Humidity AVG','Wind Speed Daily AVG','Wind Direction AVG'
                ,'Solare Radiation AVG' , 'Route_Disappointment Cleaver' ,'Route_Gibralter Ledges' ,'Route_Ingraham Direct']]


x_train.head()


y_train = s_pooled2.loc[np.isnan(s_pooled2['Battery Voltage AVG']) ==0,'Success Percentage']


y_train.head()

# Organize our data for training
X = x_train
Y = y_train
X, X_Val, Y, Y_Val = train_test_split(X, Y , train_size =0.7 , shuffle=True)

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X, Y)
pred_0=pd.DataFrame(reg.predict(X_Val)).set_index(Y_Val.index)
pred_0.columns=['pred']
 pred_0['pred']=np.where(pred_0['pred'] <0 , 0 , pred_0['pred'])

 # Print the r2 score
print(r2_score(Y_Val, np.round(pred_0['pred'],0)))
# Print the mse score
print(mean_squared_error(Y_Val,np.round(pred_0['pred'],0)))


# Organize our data for training
X = x_train
Y = y_train
X, X_Val, Y, Y_Val = train_test_split(X, Y , train_size =0.7 , shuffle=True)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# A parameter grid for XGBoost
params = {'learning_rate': [ 0.01, 0.03 , 0.05 ,0.1] , 'n_estimators' :[100,  300 , 500 , 800 , 1000] , 'max_depth' : [i for i in range(8)]
}

# Initialize XGB and GridSearch
xgb = XGBRegressor(nthread=-1 , objective = 'reg:squarederror')

grid1 = GridSearchCV(xgb, params)
grid1.fit(X, Y)


# Print the r2 score
print(r2_score(Y_Val, grid1.best_estimator_.predict(X_Val)))
# Print the mse score
print(mean_squared_error(Y_Val, grid1.best_estimator_.predict(X_Val)))


grid1.best_estimator_

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


params = { 'learning_rate' : [0.01] , 'n_estimators':[500] , 'subsample' : [i/10.0 for i in range(11)] ,'min_child_weight' :[0.5 , 1 , 1.5 , 2] , 'colsample_bytree' : [i / 10.0 for i in range(11)]}


# Initialize XGB and GridSearch
xgb = XGBRegressor(nthread=-1 , objective = 'reg:squarederror')

grid2 = GridSearchCV(xgb, params)
grid2.fit(X, Y)


# Print the r2 score
print(r2_score(Y_Val, grid2.best_estimator_.predict(X_Val)))
# Print the mse score
print(mean_squared_error(Y_Val, grid2.best_estimator_.predict(X_Val)))

grid2.best_estimator_
model=grid2.best_estimator_
ft = pd.Series(model.feature_importances_ , index=x_train.columns)
ft.plot(kind='bar')
