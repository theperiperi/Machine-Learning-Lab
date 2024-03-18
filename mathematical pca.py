#mathematical PCA

import numpy as np

class PCA:
    def __init__(self,n_components):
        self.n_components=n_components
        self.components=None
        self.mean=None 

    def fit(self,X):
        #mean centering
        self.mean=np.mean(X,axis=0)
        X=X-self.mean

        #covariance matrix
        cov_matrix=np.cov(X.T)

        #eigen decomposition
        eigenvalues,eigenvectors=np.linalg.eig(cov_matrix)

        #sort eigenvalues and eigenvectors
        idx=np.argsort(eigenvalues)[::-1]
        eigenvectors=eigenvectors[:,idx]

        #select top k eigen vectors
        self.components=eigenvectors[:,:self.n_components]

    def transform(self,X):
        X=X-self.mean 
        return np.dot(X,self.components)
    
X=np.array([[1,2],[3,4],[5,6]])
pca=PCA(n_components=1)
pca.fit(X)
X_pca=pca.transform(X)
print("original data\n",X)
print("Transformed data\n",X_pca)