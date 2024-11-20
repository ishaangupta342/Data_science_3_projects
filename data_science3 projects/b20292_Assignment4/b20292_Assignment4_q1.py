# Name-Ishaan Gupta, Roll no.-(B20292), Mob no.-9179242114
#q1
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

df=pd.read_csv('SteelPlateFaults-2class.csv')

x=df.drop('Class',axis=1)
x_label=df['Class']
x_train,x_test,x_label_train,x_label_test= train_test_split(x,x_label,test_size=0.3,random_state=42,shuffle=True)

x_train =x_train.assign(Class=x_label_train)
x_train.to_csv('SteelPlateFaults-train.csv', index=False)
x_test =x_test.assign(Class=x_label_test)
x_test.to_csv('SteelPlateFaults-test.csv', index=False)
def q1(i) :
    model=KNeighborsClassifier(n_neighbors=i)
    model.fit(x_train,x_label_train)
    predictions=model.predict(x_test)
    print('confusion matrix for k=',i,': ')
    print(confusion_matrix(x_label_test,predictions))
    print('classification accuracy for k=',i,': ',accuracy_score(x_label_test,predictions))
    print()
    
q1(1)
q1(3)
q1(5)
