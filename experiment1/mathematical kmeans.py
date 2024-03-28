import pandas as pd
from sklearn.datasets import load_iris
import math
import random

# Load Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Convert to list of lists (required by custom_kmeans)
data = iris_df.iloc[:, :-1].values.tolist()

# Actual labels (for calculating accuracy later)
actual_labels = iris_df['target'].tolist()

def euclidean_distance(p1, p2):
    return math.sqrt(sum([(a-b)**2 for a, b in zip(p1, p2)]))

def assign_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for coordinate in data:
        distances = [euclidean_distance(coordinate, centroid) for centroid in centroids]
        cluster_index = distances.index(min(distances))
        clusters[cluster_index].append(coordinate)
    return clusters

def update_centroids(clusters):
    centroids = []
    for cluster in clusters:
        centroid = [sum(point) / len(cluster) for point in zip(*cluster)]
        centroids.append(centroid)
    return centroids

def custom_kmeans(data, n_clusters, max_iters=100):
    centroids = random.sample(data, n_clusters)

    for _ in range(max_iters):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(clusters)

        # Check if same
        if new_centroids == centroids:
            break

        centroids = new_centroids

    # Assign labels based on final centroids
    labels = []
    for coordinate in data:
        distances = [euclidean_distance(coordinate, centroid) for centroid in centroids]
        label = distances.index(min(distances))
        labels.append(label)

    return labels, centroids

# Apply custom K-means
n_clusters = 3  # Since Iris dataset has 3 classes
labels_custom, centroids_custom = custom_kmeans(data, n_clusters)

# Now let's calculate accuracy
from sklearn.metrics import accuracy_score

# Map the clusters to the actual labels
def map_cluster_to_label(cluster_labels, actual_labels):
    label_map = {}
    for cluster in set(cluster_labels):
        cluster_indices = [i for i, x in enumerate(cluster_labels) if x == cluster]
        mapped_labels = [actual_labels[i] for i in cluster_indices]
        most_common_label = max(set(mapped_labels), key=mapped_labels.count)
        label_map[cluster] = most_common_label
    return label_map

label_map = map_cluster_to_label(labels_custom, actual_labels)
predicted_labels = [label_map[label] for label in labels_custom]

# Calculate accuracy
accuracy = accuracy_score(actual_labels, predicted_labels)
print("Custom K-means Accuracy:", accuracy)
print("Centroids:", centroids_custom)