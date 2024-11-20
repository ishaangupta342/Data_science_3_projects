# Name-Ishaan Gupta, Roll no.-(B20292), Mob no.-9179242114
#q2
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

df1=pd.read_csv('SteelPlateFaults-train.csv')  
minmaxdf1=(df1-df1.min())/(df1.max()-df1.min())*(1-0)
   
df2=pd.read_csv('SteelPlateFaults-test.csv')
minmaxdf2=(df2-df2.min())/(df2.max()-df2.min())*(1-0)
    
minmaxdf1.to_csv('SteelPlateFaults-train-normalised.csv')
minmaxdf2.to_csv('SteelPlateFaults-test-normalised.csv')

df=pd.read_csv('SteelPlateFaults-2class.csv')

x=df.drop('Class',axis=1)
x_label=df['Class']
x_train,x_test,x_label_train,x_label_test= train_test_split(x,x_label,test_size=0.3,random_state=42,shuffle=True)

def q2(i) :
    model=KNeighborsClassifier(n_neighbors=i)
    model.fit(minmaxdf1.drop('Class', axis=1),x_label_train)
    predictions=model.predict(minmaxdf2.drop('Class', axis=1))
    print('confusion matrix for k=',i,': ')
    print(confusion_matrix(x_label_test,predictions))
    print('classification accuracy for k=',i,': ',accuracy_score(x_label_test,predictions))
    print()
    
q2(1)
q2(3)
q2(5)
