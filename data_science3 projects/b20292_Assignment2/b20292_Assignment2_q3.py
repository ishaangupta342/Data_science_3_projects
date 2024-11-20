# Name-Ishaan Gupta, Roll no.-(B20292), Mob no.-9179242114
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1=pd.read_csv("landslide_data3_miss.csv")
df1.dropna(subset=['stationid'], inplace=True)

df1.dropna(thresh=7,inplace=True)

p = list(df1.isnull().sum())
q = list(df1)

print('missing values in the attribute',q[0],':',p[0])
print('missing values in the attribute',q[1],':',p[1])
print('missing values in the attribute',q[2],':',p[2])
print('missing values in the attribute',q[3],':',p[3])
print('missing values in the attribute',q[4],':',p[4])
print('missing values in the attribute',q[5],':',p[5])
print('missing values in the attribute',q[6],':',p[6])
print('missing values in the attribute',q[7],':',p[7])
print('missing values in the attribute',q[8],':',p[8])

print('total number of missing values in the file: ',sum(p))

