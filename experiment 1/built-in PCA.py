import numpy as np 
from sklearn.decomposition import PCA 

#generate random sample data
np.random.seed(0)
X=np.random.rand(10,3) #100 samples with 3 features

print("data before PCA")
print(X)

def pca_builtin(X,k):
    pca=PCA(n_components=k)
    reduced_data=pca.fit_transform(X)
    return reduced_data

#example usage
k=2
reduced_data=pca_builtin(X,k)
print("data after PCA")
print(reduced_data)
