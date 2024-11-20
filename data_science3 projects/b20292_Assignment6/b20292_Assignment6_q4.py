# Name-Ishaan Gupta, Roll no.-(B20292), Mob no.-9179242114
# a6q4

import pandas as pd
import numpy as np
import math
from statsmodels.tsa.ar_model import AutoReg as AR

#reading csv
df = pd.read_csv('daily_covid_cases.csv', parse_dates=['Date'],index_col=['Date'],sep=',')
                     
test_size = 0.35                    
X = df.values
ts = math.ceil(len(X)*test_size)
train, test = X[:len(X)-ts], X[len(X)-ts:]
temp = pd.Series(X[:len(X)-ts, 0])


l=1
r=len(train)

while l <= r:
	m = int(l+(r-l)/2)
	if temp.autocorr(lag = m) > 2/np.sqrt(len(train)):
		l = m+1
	else:
		r = m-1

print('The heuristic value for the optimal number of lags is',r)

# AR model   
window=r                       
model = AR(train, lags=window)
model_fit = model.fit()                 
coef = model_fit.params                 
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = []
for t in range(len(test)):
	lag = [history[i] for i in range(len(history)-window,len(history))]
	yhat= coef[0]
	for d in range(window):
		yhat+= coef[d+1]*lag[window-d-1]           
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)

# calculate RMSE 
def rmse(a,p):
	return np.sqrt(np.sum(np.square(a-p))/(len(a)))/np.mean(a)*100

# calculate MAPE
def mape(a,p):
    a,p = np.array(a), np.array(p)
    return np.mean(np.abs((a-p)/a))*100

print('RMSE :',round(rmse(test, np.array(predictions)),3),'%')
print('MAPE :',round(mape(test, predictions),3),'%')