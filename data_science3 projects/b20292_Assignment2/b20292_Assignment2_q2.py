# Name-Ishaan Gupta, Roll no.-(B20292), Mob no.-9179242114
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1=pd.read_csv("landslide_data3_miss.csv")

x=len(df1)
print('total tuples: ',x)
df1.dropna(subset=['stationid'], inplace=True)
y=len(df1)

print('tuples deleted in step 2a: ',x-y)

df1.dropna(thresh=7,inplace=True)
z=len(df1)
print('tuples deleted in step 2b: ',y-z)



