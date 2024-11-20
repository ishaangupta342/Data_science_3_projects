# Name-Ishaan Gupta, Roll no.-(B20292), Mob no.-9179242114
# part b q2


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
l = LinearRegression()

df = pd.read_csv('abalone.csv')

x = df.loc[::, :'Shell weight']
y = df['Rings']
x_train, x_test, x_train_label, x_test_label = train_test_split(x, y,test_size=.3,random_state=42)
l.fit(x_train.values, x_train_label.values)

a1= round((1-((np.sqrt(mse(x_train_label, l.predict(x_train.values))))/x_train_label.mean()))*100,3)
print('The prediction accuracy on the training data using root mean squared error::', a1, '%')

a2 = round((1-((np.sqrt(mse(x_test_label,l.predict(x_test.values))))/x_test_label.mean()))*100,3)
print('The prediction accuracy on the testing data using root mean squared error::', a2, '%')

plt.scatter(x_train_label, l.predict(x_train.values), color='orange')
plt.title('Actual rings vs Predicted rings')
plt.xlabel('Actual no. of rings')
plt.ylabel('Predicted no. of rings')
plt.show()