import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics

df1=pd.read_csv("landslide_data3_miss.csv")
df2=pd.read_csv("landslide_data3_original.csv")
print('before changing')
for m in df1.columns:
    if m != 'dates' and m != 'stationid':
        
        print('mean of attribute', m,':',df1[m].mean())
        print('mode of attribute', m,':',df1[m].mode())
        print('median of attribute', m,':',df1[m].median())
        print('standard deviation of attribute', m,':',df1[m].std())
     
for j in df1.columns:
    if j != 'dates' and j != 'stationid':
        df1[j].fillna(df1[j].mean(),inplace= True)
        
print('')
print('after changing')
for m in df1.columns:
    if m != 'dates' and m != 'stationid':
        print('mean of attribute', m,':',df1[m].mean())
        print('mode of attribute', m,':',df1[m].mode())
        print('median of attribute', m,':',df1[m].median())
        print('standard deviation of attribute', m,':',df1[m].std())
     
    
df3=pd.read_csv("landslide_data3_miss.csv")        
def RMSE(x):
    Na = df3[x].isnull().sum()
    s=0
    for j in range(len(df1)):
        s=s+(df1[x][j]-df2[x][j])**2
    p = (s/Na)**(0.5)
    return p
arr=[]

for l in df1.columns:
    if l != 'dates' and l != 'stationid':
        print('RMSE of attribute', l,':', RMSE(l) )
        arr.append(RMSE(l))
    else:
        arr.append(0)

fig=plt.figure(figsize=(10,5))
plt.bar(df1.columns,arr,color='orange',width=0.5)
plt.show()

