# Name-Ishaan Gupta, Roll no.-(B20292), Mob no.-9179242114
# part b q4

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

l = LinearRegression()
df = pd.read_csv('abalone.csv')
err3 = []
x = df.loc[::, :'Shell weight']
y = df['Rings']
x_train, x_test, x_train_label, x_test_label = train_test_split(x, y)

for i in range(2, 6):

    poly_features = PolynomialFeatures(i)
    x_poly = poly_features.fit_transform(x_train)
    l.fit(x_poly, x_train_label.values.reshape(-1, 1))
    y_pred = l.predict(x_poly)
    err3.append(np.sqrt(mean_squared_error(x_train_label, y_pred)))
    a1 = round((1-(np.sqrt(mean_squared_error(x_train_label, y_pred))/ x_train_label.mean()))*100,3)
    print('The prediction accuracy on the training data using root mean squared error for p =',i,'is :', a1, '%')

plt.bar([2, 3, 4, 5], err3, color=['orange'])
plt.title('RMSE for diff values of P (Training data)')
plt.show()

err4 = []
print()
for i in range(2, 6):
    poly_features = PolynomialFeatures(i)
    x_poly = poly_features.fit_transform(x_test)
    l.fit(x_poly, x_test_label.values.reshape(-1, 1))
    y_pred = l.predict(x_poly)
    err4.append(np.sqrt(mean_squared_error(x_test_label, y_pred)))
    a2 = round((1-(np.sqrt(mean_squared_error(x_test_label, y_pred))/x_test_label.mean()))*100,3)
    print('The prediction accuracy on the testing data using root mean squared error for p =',i,'is :', a2, '%')

plt.bar([2, 3, 4, 5], err4, color=['orange'])
plt.title('RMSE for diff values of P (Testing data)')
plt.show()

plt.scatter(x_test_label, y_pred)
plt.title('Actual rings vs Predicted rings')
plt.xlabel('Actual no. of rings')
plt.ylabel('Predicted no. of rings')
plt.show()
