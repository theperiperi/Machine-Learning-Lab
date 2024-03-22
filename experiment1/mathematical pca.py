import numpy as np
import matplotlib.pyplot as plt

def mean(data):
    """
    Calculate the mean of a list of numbers.
    """
    return sum(data) / len(data)

def center_data(data):
    """
    Center the data by subtracting the mean from each feature.
    """
    num_samples, num_features = len(data), len(data[0])
    mean_data = [mean([data[i][j] for i in range(num_samples)]) for j in range(num_features)]
    
    centered_data = [[data[i][j] - mean_data[j] for j in range(num_features)] for i in range(num_samples)]
    
    return centered_data, mean_data

def dot_product(vector1, vector2):
    """
    Calculate the dot product of two vectors.
    """
    return sum([vector1[i] * vector2[i] for i in range(len(vector1))])

def multiply_matrix_vector(matrix, vector):
    """
    Multiply a matrix by a vector.
    """
    return [dot_product(matrix[i], vector) for i in range(len(matrix))]

def transpose_matrix(matrix):
    """
    Transpose a matrix.
    """
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def calculate_covariance_matrix(data):
    """
    Calculate the covariance matrix of the centered data.
    """
    num_samples = len(data)
    num_features = len(data[0])
    
    covariance_matrix = [[0] * num_features for _ in range(num_features)]
    
    for i in range(num_features):
        for j in range(i, num_features):
            covariance_matrix[i][j] = dot_product(data[i], data[j]) / (num_samples - 1)
            covariance_matrix[j][i] = covariance_matrix[i][j]
    
    return covariance_matrix

def eigenvector_of_largest_eigenvalue(matrix):
    """
    Calculate the eigenvector corresponding to the largest eigenvalue
    using the power iteration method.
    """
    num_features = len(matrix)
    
    # Initialize a random vector
    vector = [1] * num_features
    
    for _ in range(100):  # Perform 100 iterations (adjust as needed)
        new_vector = multiply_matrix_vector(matrix, vector)
        magnitude = sum([x**2 for x in new_vector])**0.5
        vector = [x / magnitude for x in new_vector]
    
    return vector

def pca(data, num_components):
    """
    Perform PCA on the data data.
    
    Parameters:
        data: Input data matrix, shape (m, n) where m is the number of samples
           and n is the number of features.
        num_components: Number of principal components to keep.
    
    Returns:
        new_data: Data transformed into the new reduced-dimensional space.
        components: The principal components.
    """
    centered_data, mean_data = center_data(data)
    
    # Calculate the covariance matrix
    covariance_matrix = calculate_covariance_matrix(centered_data)
    
    # Calculate the top 'num_components' eigenvectors
    components = [eigenvector_of_largest_eigenvalue(covariance_matrix) for _ in range(num_components)]
    
    # Project the data onto the new reduced-dimensional space
    new_data = [multiply_matrix_vector(transpose_matrix(components), sample) for sample in centered_data]
    
    return new_data, components

# Generate some random data for demonstration
np.random.seed(0)
data = np.random.rand(100, 2) * 10  # 100 samples, 2 features

# Perform PCA with 1 component
num_components = 1
transformed_data, components = pca(data.tolist(), num_components)

# Plot the original data and the principal component
plt.figure(figsize=(8, 4))
plt.scatter([sample[0] for sample in data], [sample[1] for sample in data], label='Original Data')
plt.plot([0, components[0][0] * 10], [0, components[0][1] * 10], color='red', label='Principal Component')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('PCA Example')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()

print("Principal Component(s):\n", components)
print("\nTransformed Data (First 5 rows):\n", transformed_data[:5])
