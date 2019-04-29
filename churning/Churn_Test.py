# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd

# Create a Dataframe from CSV
#my_dataframe = pd.read_csv('Customer_file_with_predictions.csv')

# Drop via logic: similar to SQL 'WHERE' clause
#my_dataframe = my_dataframe[(my_dataframe.Predictions != 0)]

#my_dataframe.to_csv('Exit_preds.csv')

file_name = 'Non_Exit_preds.csv'

# Importing the dataset
dataset = pd.read_csv(file_name)

X = dataset.iloc[:, [4, 7, 8, 9, 10, 11, 12, 13]].values 

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)



# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_


'''
# Applying Kernel PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(kernel = 'rbf', n_components = 2, fit_inverse_transform = 'True')
X = kpca.fit_transform(X)
'''



#Finding out the appropriate number of clusters
from sklearn.cluster import KMeans


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')



#X = sc.fit_transform(X)
#Applying the kmeans algorithm to dataset X
kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
ykmeans = kmeans.fit_predict(X)


X = sc.inverse_transform(X)

#Visualising the results
plt.scatter(X[ykmeans == 0, 0], X[ykmeans == 0, 1], marker = '*', s = 30, c = 'red', label = 'Careful')
plt.scatter(X[ykmeans == 1, 0], X[ykmeans == 1, 1], marker = '*', s = 30, c = 'cyan', label = 'Average')
#plt.scatter(X[ykmeans == 2, 0], X[ykmeans == 2, 1], marker = '*', s = 100, c = 'magenta', label = 'Target')
#plt.scatter(X[ykmeans == 3, 0], X[ykmeans == 3, 1], marker = '*', s = 100, c = 'yellow', label = 'Careful')
#plt.scatter(X[ykmeans == 4, 0], X[ykmeans == 4, 1], marker = '*', s = 100, c = 'blue', label = 'Sensible')
#plt.scatter(X[ykmeans == 5, 0], X[ykmeans == 5, 1], marker = '*', s = 100, c = 'green', label = 'Sensless')
#plt.scatter(X[ykmeans == 6, 0], X[ykmeans == 6, 1], marker = '*', s = 100, c = 'gray', label = 'Sense')
#plt.scatter(X[ykmeans == 7, 0], X[ykmeans == 6, 1], marker = '*', s = 30, c = 'violet', label = 'Sensless')

#plt.scatter(sc.inverse_transform(kmeans.cluster_centers_[:, 0]), sc.inverse_transform(kmeans.cluster_centers_[:, 1]), s = 300, c = 'black', label = 'Cluster Centres')

xpltvalues = []
ypltvalues = []

for x in range(1000):
    if x >= 300:
        xpltvalues.append(x)
        ypltvalues.append(50)
    
plt.plot(xpltvalues, ypltvalues, color = 'black')



plt.title('Clusters of Clients who will stay at the bank prob 0.49 and below')
plt.xlabel('Credit Score')
plt.ylabel('Age')
#plt.legend()
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('Exit probability between 0 and 0.5.png', dpi=300)


#PLOTTING COMPARISION
x1values = [0, 0.25, 0.5, 0.75]
xvalues = [0.125, 0.375, 0.625, 0.875]

yfiftyplus = [564, 312, 234, 285]
yfiftyminus = [6946, 1037, 424, 198]

plt.plot(xvalues, yfiftyplus, label = 'Above 50',color = 'red')
plt.plot(xvalues, yfiftyminus, label = 'Below 50',color = 'blue')

plt.title('Comparing people above 50 and those below for the diff probabilites')
plt.xlabel('Exit probability')
plt.ylabel('Number of People')
plt.legend()
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('Comparing people above 50 and those below for the diff probabilites.png', dpi=300)

#PLOTING THE PERCENTAGES
yfiftyplusperc = []
yfiftyminusperc = []


for n in range(4):
    print(n)    
    yfiftyplusperc.append((yfiftyplus[n]*100)/(yfiftyplus[n]+yfiftyminus[n]))   
    yfiftyminusperc.append((yfiftyminus[n]*100)/(yfiftyplus[n]+yfiftyminus[n]))


plt.plot(xvalues, yfiftyplusperc, label = 'Above 50',color = 'red')
plt.plot(xvalues, yfiftyminusperc, label = 'Below 50',color = 'blue')

plt.title('Comparing people above 50 and those below for the diff probabilites')
plt.xlabel('Exit probability')
plt.ylabel('Number of People')
plt.legend()
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('Percentage changes of the different probabilites.png', dpi=300)
    



