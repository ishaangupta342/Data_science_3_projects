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
#q3a

# defining function to find standard deviation
def standev(m):
    return((((len(a))/(len(a)-1))**0.5)*np.std(m))

# defining function to find correlation coefficient
def corr_coeff(x,y):
    n = 0
    for i in range(len(x)):
        n += (x[i]-np.average(x))*(y[i]-np.average(y))
    covariance = n/(len(a)-1)
    z=covariance/(standev(x)*standev(y))
    return(z)

print('Correlation coefficient between Age and pregs:',corr_coeff(h,a))
print()
print('Correlation coefficient between Age and plas:',corr_coeff(h,b))
print()
print('Correlation coefficient between Age and pres:',corr_coeff(h,c))
print()
print('Correlation coefficient between Age and skin:',corr_coeff(h,d))
print()
print('Correlation coefficient between Age and test:',corr_coeff(h,e))
print()
print('Correlation coefficient between Age and BMI:',corr_coeff(h,f))
print()
print('Correlation coefficient between Age and pedi:',corr_coeff(h,g))
print()
print('Correlation coefficient between Age and Age:',corr_coeff(h,h))
print()


#q3b

print('Correlation coefficient between BMI and pregs:',corr_coeff(f,a))
print()
print('Correlation coefficient between BMI and plas:',corr_coeff(f,b))
print()
print('Correlation coefficient between BMI and pres:',corr_coeff(f,c))
print()
print('Correlation coefficient between BMI and skin:',corr_coeff(f,d))
print()
print('Correlation coefficient between BMI and test:',corr_coeff(f,e))
print()
print('Correlation coefficient between BMI and BMI:',corr_coeff(f,f))
print()
print('Correlation coefficient between BMI and pedi:',corr_coeff(f,g))
print()
print('Correlation coefficient between BMI and Age:',corr_coeff(f,h))


