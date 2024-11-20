# Name-Ishaan Gupta, Roll no.-(B20292), Mob no.-9179242114
# part a q1

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
import warnings

warnings.filterwarnings('ignore')
df_train = pd.read_csv('SteelPlateFaults-train.csv')
df_test = pd.read_csv('SteelPlateFaults-test.csv')

drop_att=['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400']
for att in drop_att:
    df_train.drop(att,axis=1,inplace = True)
    df_test.drop(att,axis=1,inplace = True)
    
x_train = df_train.drop('Class', axis= 1)
x_train_label = df_train['Class']
x_test = df_test.drop('Class', axis= 1)
x_test_label = df_test['Class']

class0 = df_train.groupby('Class').get_group(0)
class0.drop('Class',axis=1 ,inplace=True)
class1 = df_train.groupby('Class').get_group(1)
class1.drop('Class',axis=1 ,inplace=True)

def pAq1(Q):
    GMM = GaussianMixture(n_components=Q, covariance_type='full', random_state= 42)
    GMM.fit(class0)
    p0=GMM.score_samples(x_test)+np.log(len(class0)/len(df_train))
    GMM.fit(class1)
    p1=GMM.score_samples(x_test)+np.log(len(class1)/len(df_train))
    predicted_class= []
    for i in range(len(x_test)):
        if p0[i]>p1[i]:
            predicted_class.append(0)
        else:
            predicted_class.append(1)
    print('Confusion matrix for value of Q=',Q,':')
    print(confusion_matrix(x_test_label,predicted_class))
    print('classification accuracy for value of Q=',Q,':',accuracy_score(x_test_label,predicted_class))
    print()
    
pAq1(2)
pAq1(4)
pAq1(8)
pAq1(16)
    
    
    
    
    
    
    