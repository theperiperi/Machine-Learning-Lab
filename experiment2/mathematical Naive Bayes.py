import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probabilities=None
        self.feature_probabilities=None 

    def fit(self,X,y):
        #calculate class probabilities
        self.class_probabilities={label:np.mean(y==label) for label in np.unique(y)}

        #calculate feature probabilities
        self.feature_probabilities={}
        for label in self.class_probabilities:
            label_indices=np.where(y==label)
            class_features=X[label_indices]
            self.feature_probabilities[label]={
                "mean":np.mean(class_features,axis=0),
                "std":np.std(class_features,axis=0)
            }