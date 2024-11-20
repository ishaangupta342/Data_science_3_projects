# Name-Ishaan Gupta, Roll no.-(B20292), Mob no.-9179242114
#q3
import pandas as pd
import numpy as np
from numpy.linalg import det, inv
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df_train=pd.read_csv('SteelPlateFaults-train.csv')
df_test=pd.read_csv('SteelPlateFaults-test.csv')

drop_att=['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400']
for att in drop_att:
    df_train.drop(att,axis=1,inplace = True)
    df_test.drop(att,axis=1,inplace = True)

df1=df_train[df_train['Class'] == 1].copy()
df1.drop('Class',axis=1,inplace = True)
df0=df_train[df_train['Class'] == 0].copy()
df0.drop('Class',axis=1,inplace = True)

u1=df1.mean()
u0=df0.mean()
pd.DataFrame(u0).to_csv('Mean_vector_for_class_0.csv')
pd.DataFrame(u1).to_csv('Mean_vector_for_class_1.csv')
df0.cov().to_csv('Covariance_matrix_for_class_0.csv')
df1.cov().to_csv('Covariance_matrix_for_class_1.csv')

def P(x,i,u,P_Ci):
    A=x-u
    p=np.dot(A.T,i)
    pf= np.dot(p,A)
    return np.log(P_Ci)+0.5*np.log(det(i))-11.5*np.log(2*np.pi)-0.5*pf

predicted_class=[]
for k in df_test.index:
    P0=P(df_test.iloc[k, :23],inv(df0.cov()),u0,273/(273+509))
    P1=P(df_test.iloc[k, :23],inv(df1.cov()),u1,509/(273+509))
    if P0>P1:
        predicted_class.append(0)
    else:
        predicted_class.append(1)

print('confusion matrix :')
print(confusion_matrix(df_test['Class'],predicted_class))
print('classification accuracy :')
print(accuracy_score(df_test['Class'],predicted_class))