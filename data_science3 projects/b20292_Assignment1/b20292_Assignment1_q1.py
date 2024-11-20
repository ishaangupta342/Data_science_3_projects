# Name-Ishaan Gupta, Roll no.-(B20292), Mob no.-9179242114
import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt

df=pd.DataFrame(pd.read_csv('pima-indians-diabetes.csv'))
# making lists
a=list(df['pregs'])
b=list(df['plas'])
c=list(df['pres'])
d=list(df['skin'])
e=list(df['test'])
f=list(df['BMI'])
g=list(df['pedi'])
h=list(df['Age'])
i=list(df['class'])

#q1 defining function to find data for different attributes 
def part_1(x):
    print('mean-',statistics.mean(x))
    print('median-',statistics.median(x))
    print('mode-',statistics.mode(x))
    print('minimum-',min(x))
    print('maximum-',max(x))
    print('standard deviation-',statistics.stdev(x))

print('data for attribute pregs'),part_1(a)
print()
print('data for attribute plas'),part_1(b)
print()
print('data for attribute pres'),part_1(c)
print() 
print('data for attribute skin'),part_1(d)
print()
print('data for attribute test'),part_1(e)
print()
print('data for attribute BMI'),part_1(f)
print()
print('data for attribute pedi'),part_1(g)
print()
print('data for attribute Age'),part_1(h)

