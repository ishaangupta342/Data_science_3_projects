# Name-Ishaan Gupta, Roll no.-(B20292), Mob no.-9179242114
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1=pd.read_csv("landslide_data3_miss.csv")

for j in df1.columns:
    if j != 'dates' and j != 'stationid':
        df1[j].interpolate(method='linear',inplace= True)

plt.boxplot(df1['temperature'])
plt.show()


plt.boxplot(df1['rain'])
plt.show()
