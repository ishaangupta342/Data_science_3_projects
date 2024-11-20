# Name-Ishaan Gupta, Roll no.-(B20292), Mob no.-9179242114
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1=pd.read_csv("landslide_data3_miss.csv")

for j in df1.columns:
    if j != 'dates' and j != 'stationid':
        df1[j].interpolate(method='linear',inplace= True)

def outlier(x):
    q1=np.percentile(df1[x],25)     
    q3=np.percentile(df1[x],75)   
    max=q3+1.5*(q3 - q1)          
    min=q1-1.5*(q3 - q1)         
    q1out=df1[df1[x]<min]               
    q3out=df1[df1[x]>max]      
    for m in q1out[x].index:         
        df1[x].loc[m]=df1[x].median()
    for k in q3out[x].index:         
        df1[x].loc[k]=df1[x].median()
        
outlier('rain')
outlier('temperature')

plt.boxplot(df1['temperature'])
plt.show()

plt.boxplot(df1['rain'])
plt.show()

