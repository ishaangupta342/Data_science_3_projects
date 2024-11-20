# Name-Ishaan Gupta, Roll no.-(B20292), Mob no.-9179242114
#q1
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
df=pd.read_csv('pima-indians-diabetes.csv')
df1=df.copy()
df1.drop(['class'],axis=1,inplace=True)

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
        
for i in df.columns:
    if i != 'class' :
        outlier(i)
    
def minmaxnormalization(x,y):
    data={'Before Normalisation':[min(x),max(x)],'After Normalisation':[min(y),max(y)]}
    df2=pd.DataFrame(data,index=['Minimum','Maximum'])
    print(df2)
    
minmaxdf=(df1-df1.min())/(df1.max()-df1.min())*(12-5)+5
for k in minmaxdf.columns: 
        print('Min-max normilization for',k,':')
        minmaxnormalization(df1[k],minmaxdf[k])
        
print()
def standardization(x,y):
    data={'Before Normalisation':['%.3f'%(x.mean()),'%.3f'%x.std()],'After Normalisation':['%.3f'%(y.mean()),'%.3f'%y.std()]}
    df3=pd.DataFrame(data,index=['Mean','Std'])
    print(df3)

zdf=(df1-df1.mean())/(df1.std())
for l in zdf.columns:
        print('Standardization for',l,':')
        standardization(df1[l],zdf[l])

