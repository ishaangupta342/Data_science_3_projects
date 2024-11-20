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

#q6 plotting boxplots of different attributes

plt.boxplot(a)
plt.show()
plt.boxplot(b)
plt.show()
plt.boxplot(c)
plt.show()
plt.boxplot(d)
plt.show()
plt.boxplot(e)
plt.show()
plt.boxplot(f)
plt.show()
plt.boxplot(g)
plt.show()
plt.boxplot(h)
plt.show()
