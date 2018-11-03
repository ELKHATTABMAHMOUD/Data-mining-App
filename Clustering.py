# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 19:25:11 2018

@author: BRAHIM
"""
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import seaborn as sns


eurostat = pd.read_csv('eurostat-2013.csv', sep='\s*,\s*',header=0, encoding='', engine='python')
##print(eurostat)
#


eurostat['balance/population'] = np.where(eurostat['tec00115 (2013)'] > 1, 
eurostat['tec00115 (2013)'], 
eurostat['tec00115 (2013)']*1./eurostat['tps00001 (2013)'])
eurostat['chercheurs/population'] = np.where(eurostat['tec00115 (2013)'] > 1,
        eurostat['tec00115 (2013)'], eurostat['tec00115 (2013)']*1./eurostat['tps00001 (2013)'])
##print(eurostat)

eurostat=eurostat.drop('tps00001 (2013)', axis=1)
##print(eurostat)

print("Voila!")
#eurostat.info()

df = pd.DataFrame(eurostat)

##print(eurostat.columns.tolist())
 
df = df[["tec00115 (2013)","teilm (F dec 2013)","teilm (M dec 2013)","tec00118 (2013)",
       "teimf00118 (dec 2013)","tsdsc260(2013)","tet00002 (2013)","tsc00001 (2011)",
        "tsc00004 (2012)","balance/population","chercheurs/population"]].astype(float)
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
print(scaler.fit(df))
StandardScaler(copy=True, with_mean=True, with_std=True)
print(scaler.mean_)

x= scaler.fit_transform(df)
print("Standarization fit_transform")
print(x)
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2',
                          'principal component 3','principal component 4'])
finalDf = pd.concat([principalDf, eurostat[['Code']]], axis = 1)

print(finalDf)


variance= pca.explained_variance_ratio_

print(variance)

pca = PCA().fit(x)
plt.plot(np.cumsum(variance))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

df_variance = pd.DataFrame({'var':variance,
             'principalComponents':['principal component 1','principal component 2',
                                    'principal component 3','principal component 4']})
sns.barplot(x='principalComponents',y='var', 
           data=df_variance, color="c");
##### KMEANS
for k in range (2, 9):
 
	# Create a kmeans model on our data, using k clusters.  random_state helps ensure that the algorithm returns the same results each time.
	kmeans_model = KMeans(n_clusters=k, random_state=1).fit(df.iloc[:, :])
	
	# These are our fitted labels for clusters -- the first cluster has label 0, and the second has label 1.
	labels = kmeans_model.labels_
 
	# Sum of distances of samples to their closest cluster center
	interia = kmeans_model.inertia_
	print ("k:",k, " cost:", interia)


X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50);


df_kmeans = KMeans(n_clusters=4, random_state=0).fit(df)
y_kmeans = df_kmeans.predict(df)

#plt.scatter(df_kmeans.values[:, 0], df.values[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = df_kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);


kmeans_model2 = KMeans(n_clusters=2, random_state=1).fit(df.iloc[:, :])

Z = linkage(df, 'ward')
c, coph_dists = cophenet(Z, pdist(df))
'''

idxs = [33, 68, 62]
plt.figure(figsize=(10, 8))
plt.scatter(df.values[:,0], df.values[:,1])  # plot all points
plt.scatter(df.values[idxs,1], df.values[idxs,1], c='r')  # plot interesting points in red again
plt.show()
'''

plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()






'''
#fig = plt.figure(figsize = (4,4))
#ax = fig.add_subplot(1,1,1) 
#ax.set_xlabel('Principal Component 1', fontsize = 15)
#ax.set_ylabel('Principal Component 2', fontsize = 15)
##ax.set_title('2 component PCA', fontsize = 20)
#targets = ['US', 'UK', 'FR']
#colors = ['r', 'g', 'b']
#for target, color in zip(targets,colors):
#    indicesToKeep = finalDf['Code'] == target
#    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#               , finalDf.loc[indicesToKeep, 'principal component 2']
#               , c = color
#               , s = 50)
#ax.legend(targets)
#ax.grid()
#☻pca2 = PCA(n_components=3)

#principalDf2 = pd.DataFrame(data = principalComponents
 #            , columns = ['principal factor 1', 'principal factor 2','principal factor 3'])
#finalDf2 = pd.concat([principalDf2, eurostat[['Code']]], axis = 1)

#print(finalDf2)
#df.head()
##df.info()

##print(eurostat['tec00115 (2013)'])


#☺df = pd.DataFrame({'var':pca.explained_variance_ratio_,
#             'PC':["tec00115 (2013)","teilm (F dec 2013)","teilm (M dec 2013)","tec00118 (2013)",
 #      "teimf00118 (dec 2013)","tsdsc260(2013)","tet00002 (2013)","tsc00001 (2011)",
  #      "tsc00004 (2012)","balance/population","chercheurs/population"]})
'''

