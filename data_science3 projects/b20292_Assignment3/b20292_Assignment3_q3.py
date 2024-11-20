# Name-Ishaan Gupta, Roll no.-(B20292), Mob no.-9179242114
#q3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#a
df=pd.read_csv("pima-indians-diabetes.csv")
df1=df.copy()
df1.drop(['class'],axis=1,inplace=True)
zdf=(df1-df1.mean())/(df1.std()) 

eigval,eigvec=np.linalg.eig(np.cov(zdf.T))
eigval.sort()
eigval=eigval[::-1]

pca=PCA(n_components=2)
principalComponents = pca.fit_transform(zdf)
principalDf = pd.DataFrame(data=principalComponents,columns=['principal component 1', 'principal component 2'])
print(principalDf.cov().values)
eigenvalue,eigenvector=np.linalg.eig(np.array([principalDf.cov().values]))
print(eigenvector[0])
plt.quiver([0,0],[0,0],eigenvector[0][0],eigenvector[0][1],scale=4)
for i in range(1,3):
    print('Variance along Eigen Vector',i,':',np.var(principalComponents.T[i-1]))
    print('Eigen Value corresponding to Eigen Vector',i,':',eigval[i-1])


plt.figure(figsize=(10,6))
plt.scatter(principalDf['principal component 1'],principalDf['principal component 2'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title("Scatter plot of reduced dimensional data")
plt.show()


#b
plt.figure(figsize=(10,6))
plt.bar(range(1,9),eigval)
plt.scatter(range(1,9),eigval)
plt.plot(range(1,9),eigval,color='red')
plt.xlabel('L')
plt.ylabel('Eigenvalues')
plt.title('Plot of Eigenvalues')
plt.show()

#c
rmse_list=[]
for i in range(1,9):
    pca=PCA(n_components=i)    
    principalComponents = pca.fit_transform(zdf)
    approx_data = pca.inverse_transform(principalComponents)  
    dfpca = pd.DataFrame(principalComponents,columns=list(range(i)))
    covmatrix = dfpca.cov()
    rmse=(np.square(np.subtract(zdf.values,approx_data)).mean())**.5 
    rmse_list.append(rmse)
    print(covmatrix.to_csv())
    print()
plt.figure(figsize=(10,6))
plt.bar(range(1,9),rmse_list)
plt.scatter(range(1,9),rmse_list,color='black')
plt.plot(range(1,9),rmse_list,color='red')
plt.xlabel('L')
plt.ylabel('RMSE')
plt.title('Plot of RMSE')
plt.show()

#d
print(covmatrix)
print(zdf.cov().to_csv())


