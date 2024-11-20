# Name-Ishaan Gupta, Roll no.-(B20292), Mob no.-9179242114
# q6

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn import metrics, decomposition
from sklearn.cluster import DBSCAN

# reading csv file
df = pd.read_csv('Iris.csv') 
df_data = df.iloc[:,:-1]
df_label = df['Species']
df_pca = decomposition.PCA(n_components=2).fit_transform(df_data)
df_pca = pd.DataFrame(df_pca, columns=['D1', 'D2'])

def purity_score(y_true, y_pred): 
    # compute contingency matrix (also called confusion matrix) 
    contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
    # print('Contingency matrix/Confusion matrix:',contingency_matrix)
    # Find optimal one-to-one mapping between cluster labels and true labels 
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    # Return cluster accuracy 
    return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)

# epsilon different values
eps=[1,5]  
min_samples = [4,10]  
for i in eps:
    for j in min_samples:
        # DBSCAN Model
        dbscan_model = DBSCAN(eps=i, min_samples=j) 
        dbscan_model.fit(df_pca) 
        # labels predicted after DBSCAN clustering
        DBSCAN_predictions = dbscan_model.labels_ 
        labels=np.unique(DBSCAN_predictions)
        class_labels = pd.DataFrame(DBSCAN_predictions, columns=['Class'])
        combined = pd.concat([df_pca, class_labels], axis=1)
        # plotting each cluster
        for k in labels:
            label = 'Class' + str(k)
            plt.scatter(combined.loc[combined['Class'] == k]['D1'], combined.loc[combined['Class'] == k]['D2'],
                        label=label)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title(f'DBSCAN Clustering for eps : {i} and min samples : {j}')
        plt.legend()
        plt.show()
        pur_score = purity_score(df_label, DBSCAN_predictions)
        print('The purity score for eps',i,'and min samples',j,'is :',round(pur_score, 3))