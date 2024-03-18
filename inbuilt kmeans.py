import numpy as np
from sklearn.cluster import KMeans

def custom_kmeans(X,n_clusters):
    kmeans=KMeans(n_clusters=n_clusters)
    labels=kmeans.fit_predict(X)
    centroids=kmeans.cluster_centers_

    return labels,centroids

X = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])
n_clusters = 2

labels,centroids=custom_kmeans(X,n_clusters)
print("Labels: ",labels)
print("centroids: ",centroids)

