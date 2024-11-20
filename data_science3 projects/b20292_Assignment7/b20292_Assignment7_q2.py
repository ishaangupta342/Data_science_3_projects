# Name-Ishaan Gupta, Roll no.-(B20292), Mob no.-9179242114
# q2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn import  decomposition, metrics
from sklearn.cluster import KMeans

# reading csv file
df = pd.read_csv('Iris.csv') 
df_data = df.iloc[:,:-1]
df_label = df['Species']
df_pca = decomposition.PCA(n_components=2).fit_transform(df_data)
df_pca = pd.DataFrame(df_pca, columns=['D1','D2'])

K = 3 
kmeans=KMeans(n_clusters=K) 
kmeans.fit(df_pca)
kmeans_prediction = kmeans.predict(df_pca)
clustercentre_kmeans = pd.DataFrame(kmeans.cluster_centers_) 
labels = np.unique(kmeans_prediction)  
class_labels = pd.DataFrame(kmeans_prediction, columns=['Class'])
combined = pd.concat([df_pca, class_labels], axis=1)

for i in labels:
    label = 'Class' + str(i)
    plt.scatter(combined.loc[combined['Class']==i]['D1'], combined.loc[combined['Class'] == i]['D2'], label=label)
plt.legend()
plt.scatter(clustercentre_kmeans[0], clustercentre_kmeans[1], s=70, marker='*', color='red',label = 'Centroids')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Kmeans Clustering for Reduced Data')
plt.legend()
plt.show()

def purity_score(y_true, y_pred): 
    # compute contingency matrix (also called confusion matrix) 
    contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
    # print('Contingency matrix/Confusion matrix:',contingency_matrix)
    # Find optimal one-to-one mapping between cluster labels and true labels 
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    # Return cluster accuracy 
    return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)

print(f'Distortion Measure for {K} clusters = ', round(kmeans.inertia_, 2))

print('Purity score for training data - ', round(purity_score(df_label, kmeans_prediction), 2))