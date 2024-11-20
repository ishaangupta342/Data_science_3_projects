# Name-Ishaan Gupta, Roll no.-(B20292), Mob no.-9179242114
# a6q3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg as AR


df=pd.read_csv('daily_covid_cases.csv')
l=len(df['new_cases'])
train_data_size = int(l*0.65)

train_data = df.iloc[:train_data_size, 1]
test_data = df.iloc[train_data_size:, 1]

# calculate MAPE
def mape(a,p):
    a,p = np.array(a), np.array(p)
    return np.mean(np.abs((a-p)/a))*100

# calculate RMSE 
def rmse(a,p):
	return np.sqrt(np.sum(np.square(a-p))/(len(a)))/np.mean(a)*100


RMSE=[]
MAPE=[]

# AR model
for j in [1, 5, 10, 15, 25]:
    model = AR(train_data, lags=j)
    model_fit = model.fit()
    coef=model_fit.params
    history=list(train_data[len(train_data)-j:])
    predictions = list()
    for t in range(train_data_size, l):
        lag=[history[i] for i in range(len(history)-j, len(history))]
        yhat=coef[0]
        for d in range(j):
            yhat += coef[d+1]*lag[j-d-1]
        obs=test_data[t]
        predictions.append(yhat)
        history.append(obs)
    r=rmse(test_data, predictions)
    RMSE.append(r)
    print('RMSE for test_data data for lag',j,'is',round(r,3),'%')
    m=mape(test_data, predictions)
    MAPE.append(m)
    print('MAPE for test_data data for lag',j,'is',round(m,3),'%')
    print()

plt.bar(['1', '5', '10', '15', '25'], RMSE)
plt.title('Barchart of RMSE')
plt.ylabel('RMSE')
plt.show()

plt.bar(['1', '5', '10', '15', '25'], MAPE)
plt.title('Barchart of  MAPE')
plt.ylabel('MAPE')
plt.show()