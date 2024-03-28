import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def mean(data):
    return sum(data) / len(data)

def center_data(data):
    num_samples, num_features = len(data), len(data[0])
    mean_data = [mean([data[i][j] for i in range(num_samples)]) for j in range(num_features)]
    centered_data = [[data[i][j] - mean_data[j] for j in range(num_features)] for i in range(num_samples)]
    return centered_data, mean_data

def dot_product(vector1, vector2):
    return sum([vector1[i] * vector2[i] for i in range(len(vector1))])

def multiply_matrix_vector(matrix, vector):
    return [dot_product(matrix[i], vector) for i in range(len(matrix))]

def transpose_matrix(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def calculate_covariance_matrix(data):
    num_samples = len(data)
    num_features = len(data[0])
    covariance_matrix = [[0] * num_features for _ in range(num_features)]
    for i in range(num_features):
        for j in range(i, num_features):
            covariance_matrix[i][j] = dot_product(data[i], data[j]) / (num_samples - 1)
            covariance_matrix[j][i] = covariance_matrix[i][j]
    return covariance_matrix

def eigenvector_of_largest_eigenvalue(matrix):
    num_features = len(matrix)
    vector = [1] * num_features
    for _ in range(100):  # Perform 100 iterations (adjust as needed)
        new_vector = multiply_matrix_vector(matrix, vector)
        magnitude = sum([x**2 for x in new_vector])**0.5
        vector = [x / magnitude for x in new_vector]
    return vector

def pca(data, num_components):
    centered_data, mean_data = center_data(data)
    covariance_matrix = calculate_covariance_matrix(centered_data)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    # Sort eigenvectors by eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top 'num_components' eigenvectors
    components = eigenvectors[:, :num_components]
    
    # Transform the data
    new_data = np.dot(centered_data, components)
    
    return new_data, components

# Load Iris dataset
iris = load_iris()
iris_data = iris.data.tolist()
iris_target = iris.target.tolist()

# Perform PCA with 2 components
num_components = 2
transformed_data_custom, _ = pca(iris_data, num_components)

# Convert transformed data to NumPy array
transformed_data_custom = np.array(transformed_data_custom)

# Split data into train and test sets
X_train_custom, X_test_custom, y_train_custom, y_test_custom = train_test_split(
    transformed_data_custom, iris_target, test_size=0.2, random_state=42
)

# Create a KNN classifier for custom PCA
knn_custom = KNeighborsClassifier()
knn_custom.fit(X_train_custom, y_train_custom)
y_pred_custom = knn_custom.predict(X_test_custom)
accuracy_custom = accuracy_score(y_test_custom, y_pred_custom)

# Print custom PCA accuracy
print("Custom PCA Accuracy with KNN Classifier:", accuracy_custom)
