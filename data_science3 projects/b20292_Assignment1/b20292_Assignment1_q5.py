# Name-Ishaan Gupta, Roll no.-(B20292), Mob no.-9179242114
import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt

df=pd.read_csv('pima-indians-diabetes.csv')

a=list(df['pregs'])
b=list(df['plas'])
c=list(df['pres'])
d=list(df['skin'])
e=list(df['test'])
f=list(df['BMI'])
g=list(df['pedi'])
h=list(df['Age'])
i=list(df['class'])

#q5 using groupby function to make two groups of class_ and class_1
# data = df.groupby('class')
# c0 = data.get_group(0)
# c1 = data.get_group(1)



c0= df[df['class']==0]['pregs']
c1= df[df['class']==1]

# plotting histograms with class_0 and class_1
c0.hist()
plt.title('class_0')
c1.hist('pregs')
plt.title('class_1')
plt.show()