import numpy as np 
def custom_kmeans(X,n_clusters,max_iters=100):

    #randomly initialize centroids
    centroids=X[np.random.choice(X.shape[0],size=n_clusters,replace=False)]

    for _ in range(max_iters):
        #assign each datapoint to nearest centroid
        distances= np.sqrt(((X-centroids[:,np.newaxis])**2).sum(axis=2))
        labels=np.argmin(distances,axis=0)

        #update centroids to mean of assigned data
        new_centroids=np.array([X[labels==i].mean(axis=0) for i in range(n_clusters)])

        #check for convergence
        if np.allclose(new_centroids,centroids):
            break 

        centroids=new_centroids 
    return labels,centroids

# Example usage
X = np.array([[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]])
n_clusters = 2

labels_custom,centroids_custom=custom_kmeans(X,n_clusters)
print("Labels (K-means with custom implementation):", labels_custom)
print("Centroids:")
print(centroids_custom)