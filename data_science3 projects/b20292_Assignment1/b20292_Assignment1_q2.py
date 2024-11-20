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

#q2a defining function to scatter plot attributes with age
def part_2a(x):
    plt.scatter(h,x)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.show()

part_2a(a)
part_2a(b)
part_2a(c)
part_2a(d)
part_2a(e)
part_2a(f)
part_2a(g)

#q2b defining function to scatter plot attributes with BMI
def part_2b(x):
    plt.scatter(f,x)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.show()

part_2b(a)
part_2b(b)
part_2b(c)
part_2b(d)
part_2b(e)
part_2b(g)
part_2b(h)
