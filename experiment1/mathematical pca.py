import math

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def mean_center(self, X):
        mean_vector = [sum(x) / len(x) for x in zip(*X)]
        self.mean = mean_vector
        return [[x[i] - mean_vector[i] for i in range(len(x))] for x in X]

    def cov_matrix(self, X):
        X_transposed = [list(x) for x in zip(*X)]
        cov_matrix = [[0] * len(X_transposed) for _ in range(len(X_transposed))]
        for i in range(len(X_transposed)):
            for j in range(len(X_transposed)):
                cov_matrix[i][j] = sum([(X_transposed[i][k] - self.mean[i]) * (X_transposed[j][k] - self.mean[j]) for k in range(len(X_transposed[0]))]) / (len(X_transposed[0]) - 1)
        return cov_matrix

    def eigen_decomposition(self, cov_matrix):
        eigenvalues, eigenvectors = [], []
        for i in range(len(cov_matrix)):
            b = cov_matrix[i][i]
            a = [[cov_matrix[i][j] if i == j else 0 for j in range(len(cov_matrix))] for i in range(len(cov_matrix))]
            for j in range(i + 1, len(cov_matrix)):
                c = cov_matrix[j][i]
                d = cov_matrix[i][j]
                a[i][i] = b
                a[j][j] = d
                a[i][j] = a[j][i] = c
            A = a
            lambda_matrix = [[0] * len(cov_matrix) for _ in range(len(cov_matrix))]
            for i in range(len(cov_matrix)):
                lambda_matrix[i][i] = 1
            for _ in range(100):
                Q, R = self.QR_decomposition(A)
                A = self.multiply_matrices(R, Q)
                lambda_matrix = self.multiply_matrices(lambda_matrix, Q)
            eigenvalues.append([A[i][i] for i in range(len(A))])
            eigenvectors.append([lambda_matrix[i][j] for j in range(len(lambda_matrix[0])) for i in range(len(lambda_matrix)) if j % len(lambda_matrix) == i])
        return eigenvalues[0], eigenvectors[0]

    def QR_decomposition(self, matrix):
        Q = [[0] * len(matrix) for _ in range(len(matrix))]
        R = [[0] * len(matrix) for _ in range(len(matrix))]
        for j in range(len(matrix)):
            v = [matrix[i][j] if i >= j else 0 for i in range(len(matrix))]
            mag_v = math.sqrt(sum([x ** 2 for x in v]))
            Q[j] = [v[i] / mag_v if i >= j else 0 for i in range(len(matrix))]
            for i in range(j + 1):
                R[i][j] = sum([matrix[k][j] * Q[i][k] for k in range(len(matrix))])
        return Q, R

    def multiply_matrices(self, mat1, mat2):
        result = [[0] * len(mat2[0]) for _ in range(len(mat1))]
        for i in range(len(mat1)):
            for j in range(len(mat2[0])):
                for k in range(len(mat2)):
                    result[i][j] += mat1[i][k] * mat2[k][j]
        return result

    def fit(self, X):
        # mean centering
        X_mean_centered = self.mean_center(X)

        # covariance matrix
        cov_matrix = self.cov_matrix(X_mean_centered)

        # eigen decomposition
        eigenvalues, eigenvectors = self.eigen_decomposition(cov_matrix)

        # sort eigenvalues and eigenvectors
        idx = sorted(range(len(eigenvalues)), key=lambda i: eigenvalues[i], reverse=True)
        eigenvectors = [[eigenvectors[j][i] for j in range(len(eigenvectors))] for i in idx]

        # select top k eigen vectors
        self.components = [eigenvectors[i] for i in range(self.n_components)]

    def transform(self, X):
        X_mean_centered = self.mean_center(X)
        return self.multiply_matrices(X_mean_centered, self.components)

# Example usage
X = [[1, 2], [3, 4], [5, 6]]
pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print("original data\n", X)
print("Transformed data\n", X_pca)
