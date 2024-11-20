# Name-Ishaan Gupta, Roll no.-(B20292), Mob no.-9179242114
# q1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition

# reading csv file
df = pd.read_csv('Iris.csv') 
df_data = df.iloc[:,:-1]
df_label = df['Species']

# covariance matrix for the given data
cov_mat=df_data.cov()
# determining eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_mat) 
eigenvalues = list(sorted(list(eigenvalues), reverse=True))
x=[1,2,3,4]

# plot of eigenvalues in descending order
plt.plot(x, eigenvalues)  
plt.title('EigenValues vs Components')
plt.xlabel('Components')
plt.ylabel('EigenValues')

print('EigenValues:',eigenvalues)
plt.show()
# performing PCA to reduce the dimension from 4 to 2
df_pca = decomposition.PCA(n_components=2).fit_transform(df_data)
df_pca = pd.DataFrame(df_pca, columns=['D1','D2'])
