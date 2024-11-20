# Name-Ishaan Gupta, Roll no.-(B20292), Mob no.-9179242114
#q2
#a
import numpy as np
import matplotlib.pyplot as plt
D=np.random.multivariate_normal([0,0],[[13,-3],[-3,5]],1000)
a=[D[i][0] for i in range(1000)]
b=[D[i][1] for i in range(1000)]
plt.figure(figsize=(10,6))
plt.scatter(a,b)
plt.title("Scatter plot of the data samples")
plt.show()

#b
eigenvalue,eigenvector=np.linalg.eig(np.array([[13,-3],[-3,5]]))
eigenvector1=eigenvector[:,0]
eigenvector2=eigenvector[:,1]
print(eigenvector1)
plt.figure(figsize=(10,6))
plt.scatter(a,b)
plt.quiver([0,0],[0,0],eigenvector[0],eigenvector[1],scale=4)
plt.title('Eigen directions on scatter plot of the data samples')
plt.show()

#c
m=np.dot(D,eigenvector)

plt.figure(figsize=(10,6))
plt.scatter(a,b)
plt.quiver([0,0],[0,0],eigenvector[0],eigenvector[1],scale=4)
plt.scatter(m[:,0]*eigenvector1[0],m[:,0]*eigenvector1[1],color='red')
plt.title('Projected values on eigenvector1')
plt.show()

plt.figure(figsize=(10,6))
plt.scatter(a,b)
plt.quiver([0,0],[0,0],eigenvector[0],eigenvector[1],scale=4)
plt.scatter(m[:,1]*eigenvector2[0],m[:,1]*eigenvector2[1],color='red')
plt.title('Projected values on eigenvector2')
plt.show()

#d
D_=np.dot(m,eigenvector.T)
rmse=(np.square(np.subtract(D,D_)).mean())**.5    
print('Root mean square error:',rmse)
