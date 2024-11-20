# Name-Ishaan Gupta, Roll no.-(B20292), Mob no.-9179242114
# a6q2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg as AR
import warnings

# reading csv
warnings.filterwarnings('ignore')
df = pd.read_csv('daily_covid_cases.csv')
l = len(df['new_cases'])
train_data_size = int(l*0.65)

train_data = list(df.iloc[:train_data_size, 1])
test_data = df.iloc[train_data_size:, 1]

# train_data plot
plt.plot([i for i in range(train_data_size)], train_data)
plt.title('train sets')
plt.show()

# test_data plot
plt.plot([i for i in range(train_data_size, l)], test_data)
plt.title('test sets')
plt.show()

# AR model
window=5
model = AR(train_data, lags=window)
model_fit = model.fit()   # fit/train the model
coef = model_fit.params   # Get the coefficients of AR model
print('Coefficoefients obtained from the AR model :', coef)
history = train_data[len(train_data)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()      # List to hold the predictions, 1 step at a time
for t in range(train_data_size, l):
    lag = [history[i] for i in range(len(history)-window, len(history))]
    yhat = coef[0]
    for d in range(window):
        yhat +=coef[d+1] * lag[window-d-1]  # add other values
    obs=test_data[t]
    predictions.append(yhat)   # Append predictions
    history.append(obs)        # Append actual test value to history


# q2b1
plt.scatter(test_data, predictions)
plt.xlabel('Actual no. of cases')
plt.ylabel('Predicted no. of cases')
plt.title('Scatter plot between actual and predicted values')
plt.show()

# q2b2
plt.plot(test_data, predictions)
plt.xlabel('Actual no. of cases')
plt.ylabel('Predicted no. of cases')
plt.title('Line plot between actual and predicted values')
plt.show()

# q2b3
# calculate MAPE
def mape(a,p):
    a,p = np.array(a), np.array(p)
    return np.mean(np.abs((a-p)/a))*100
print('MAPE between actual and predicted test data',round(mape(test_data, predictions),3),'%')

# calculate RMSE 
def rmse(a,p):
	return np.sqrt(np.sum(np.square(a-p))/(len(a)))/np.mean(a)*100
print('RMSE between actual and predicted test data',round(rmse(test_data, predictions),3),'%')