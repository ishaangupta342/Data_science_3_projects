# Name-Ishaan Gupta, Roll no.-(B20292), Mob no.-9179242114
# part_b q1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
l = LinearRegression()
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse 

df = pd.read_csv('abalone.csv')


pc= df.corr(method='pearson').iloc[:7, 7:]
print('Pearson correlation coefficient of target attribute with other attribute as follows :')
print(pc)

x = df['Shell weight']
y = df['Rings']
x_train, x_test, x_train_label, x_test_label = train_test_split(x, y,test_size=0.3,random_state=42)
l.fit(x_train.values.reshape(-1, 1), x_train_label.values)


plt.scatter(x_train, x_train_label, label='Train data',color='orange', marker='3', alpha=1)
plt.plot(x_train, l.predict(x_train.values.reshape(-1, 1)), label='Best-fit line', color='blue', linewidth=3)
plt.xlabel('x_train')
plt.ylabel('x_train_label')
plt.legend()
plt.show()

rmse1 = (np.sqrt(mse(x_train_label, l.predict(x_train.values.reshape(-1, 1)))))
a1 = round((1-(rmse1/x_train_label.mean()))*100,3)
print('The prediction accuracy on the training data using root mean squared error:', a1, '%')

rmse2 = (np.sqrt(mse(x_test_label, l.predict(x_test.values.reshape(-1, 1)))))
a2 = round((1-(rmse2/x_test_label.mean()))*100,3)
print('The prediction accuracy on the testing data using root mean squared error:', a2, '%')


plt.scatter(x_train_label,l.predict(x_train.values.reshape(-1, 1)), marker='*', color='orange')
plt.title('Actual rings vs Predicted rings')
plt.xlabel('Actual no. of rings')
plt.ylabel('Predicted no. of rings')
plt.show()