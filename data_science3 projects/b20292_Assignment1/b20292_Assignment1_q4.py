# Name-Ishaan Gupta, Roll no.-(B20292), Mob no.-9179242114
import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt

df=pd.DataFrame(pd.read_csv('pima-indians-diabetes.csv'))

a=list(df['pregs'])
b=list(df['plas'])
c=list(df['pres'])
d=list(df['skin'])
e=list(df['test'])
f=list(df['BMI'])
g=list(df['pedi'])
h=list(df['Age'])
i=list(df['class'])


#q4
# plotting histogram with attribute pregs
plt.hist(a)
plt.xlabel('Number of times pregnant')
plt.show()

# plotting histogram with attribute skin
plt.hist(d)
plt.xlabel('Triceps skin fold thickness (mm)')
plt.show()