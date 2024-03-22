import numpy as np
from collections import defaultdict

class NaiveBayes:
    def __init__(self):
        self.classes=None
        self.class_priors={}
        self.mean={}
        self.var={}

    def fit(self,data,y):
        self.classes=np.unique(y)
        for current_class in self.classes:
            data_current_class=data[y==current_class]
            self.class_priors[current_class]=len(data_current_class)/len(data)
            self.var[current_class]=np.var(data_current_class,axis=0)
            self.mean[current_class]=np.mean(data_current_class,axis=0)
    
    def predict(self,data):
        predictions=[]
        for coordinate in data:
            posteriors=[]
            for current_class in self.classes:
                prior=np.log(self.class_priors[current_class])
                likelihood=np.sum(np.log(self.gaussian_pdf(coordinate,self.mean[current_class],self.var[current_class])))
                posterior=prior+likelihood
                posteriors.append(posterior)
            predicted_class=self.classes[np.argmax(posteriors)]
            predictions.append(predicted_class)
        return predictions
    
    def gaussian_pdf(self, data, mean, var):
        return (1/np.sqrt(2*np.pi*var))*np.exp(-((data-mean)**2)/(2*var))
    
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5],
                    [2, 1], [3, 2], [4, 3], [5, 4]])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])
X_test = np.array([[1.5, 2.5], [4.5, 3.5]])

nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)
print("Predictions:", predictions)
    