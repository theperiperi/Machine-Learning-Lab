import math
import random 

def euclidean_distance(p1,p2):
    return math.sqrt(sum([(a-b)**2 for a,b in zip(p1,p2)]))

def assign_clusters(data,centroids):
    clusters=[[] for i in range(len(centroids))]
    for coordinate in data:
        distances=[euclidean_distance(coordinate,centroid) for centroid in centroids]
        cluster_index=distances.index(min(distances))
        clusters[cluster_index].append(coordinate)
    return clusters

def update_centroids(clusters):
    centroids=[]
    for cluster in clusters:
        centroid=[sum(point)/len(cluster) for point in zip(*cluster)]
        centroids.append(centroid)
    return centroids

def custom_kmeans(data,n_clusters,max_iters=100):
    centroids=random.sample(list(data),n_clusters)

    for i in range(max_iters):
        clusters=assign_clusters(data,centroids)
        new_centroids=update_centroids(clusters)

        #check if same
        if new_centroids==centroids:
            break 

        centroids=new_centroids 
    
    #assign labels based on final centroids
    labels=[]
    for coordinate in data:
        distances=[euclidean_distance(coordinate,centroid) for centroid in centroids]
        label=distances.index(min(distances))
        labels.append(label)
        
    return labels,centroids

#example usage
data = [[1, 2], [5, 8], [1.5, 1.8], [8, 8], [1, 0.6], [9, 11]]
n_clusters=2

labels_custom,centroids_custom=custom_kmeans(data,n_clusters)
print("custom labels are",labels_custom)
print("centroids are",centroids_custom)
            