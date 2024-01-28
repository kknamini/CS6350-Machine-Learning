import decisiontree as dt
import numpy as np
import math

class AdaboostTrees:
    def __init__(self, classifier = dt.DecisionTree(split_metric= 'gini', max_depth=1), n_classifiers = 10):
        self.classifier = classifier
        self.n_classifiers = n_classifiers
        self.votes = []
        self.ensemble = []
    
    def fit(self, X, y, attrs):
        self.ensemble = []
        self.votes = []
        
        n_examples = len(X)
        
        D = np.full(n_examples, 1/n_examples)
        
        for i in range(self.n_classifiers):
            
            stump = self.classifier
            stump.fit(X, y, attrs, list(D))
            
            predict_accuracy = []
            
            for index in range(n_examples):
                predict_accuracy.append(D[index] * (stump._predict_one(X[index], attrs) == y[index]))
            
            vote = np.log(sum(predict_accuracy) / (1 - sum(predict_accuracy))) / 2
            
            for index in range(n_examples):
                D[index] *= np.exp(-vote * y[index] * stump._predict_one(X[index], attrs))
            
            D = D / sum(D)
            
            self.ensemble.append(stump)
            self.votes.append(vote)
        
    def _predict_one(self, X, attrs):
        pred = 0
    
        for i in range(self.n_classifiers):
        
            pred += self.votes[i] * self.ensemble[i]._predict_one(X, attrs)
            
            votes_completed = sum(self.votes[:i+1])
            votes_remaining = sum(self.votes[i+1:])
            
            # Stop prediction procedure early when the current prediction is guaranteed
            # If the magnitude of the summed prediction is larger than the number of predictions remaining, then it is guaranteed to be its current sign
            # This process starts only after half the votes (each vote is not 1) have made a prediction.
            if votes_completed > votes_remaining:
                if abs(pred) > votes_remaining:
                    break
        
        return int(math.copysign(1, pred))

    
    def predict(self, X, attrs):
        preds = []
        
        for i in range(len(X)):
            preds.append(self._predict_one(X[i], attrs))
        
        preds = np.array(preds)
        
        return preds
    
class BaggedTrees:
    def __init__(self, classifier = dt.DecisionTree(), n_classifiers = 10):
        self.classifier = classifier
        self.n_classifiers = n_classifiers
        self.ensemble = []
        
    def fit(self, X, y, attrs):
        self.ensemble = []
        
        n_samples = len(X)
        
        for i in range(self.n_classifiers):
            
            indeces = np.random.randint(0, n_samples, n_samples)
            
            bootstrap_X = X[indeces,:]
            bootstrap_y = y[indeces]
            
            model = self.classifier
            
            model.fit(bootstrap_X, bootstrap_y, attrs)
            
            self.ensemble.append(model)
            
    def _predict_one(self, X, attrs):
        
        pred = 0
        
        for i in range(self.n_classifiers):
            
            pred += self.ensemble[i]._predict_one(X, attrs)
            
            # Stop prediction procedure early when the current prediction is guaranteed
            # If the magnitude of the summed prediction is larger than the number of predictions remaining, then it is guaranteed to be its current sign
            # This process starts only after half the ensemble has made a prediction vote.
            if i >= (self.n_classifiers / 2):
                if abs(pred) > self.n_classifiers - i - 1:
                    break
        
        return int(math.copysign(1, pred))

    
    def predict(self, X, attrs):
        preds = []
        
        for i in range(len(X)):
            
            preds.append(self._predict_one(X[i], attrs))
        
        preds = np.array(preds)
        
        return preds

class RandomForest:
    def __init__(self, classifier = dt.DecisionTree(rand_tree=True), n_classifiers = 10):
        self.classifier = classifier
        self.n_classifiers = n_classifiers
        self.ensemble = []
        
    def fit(self, X, y, attrs):
        self.ensemble = []
        
        for i in range(self.n_classifiers):
            
            n_samples = len(X)
            
            indeces = np.random.randint(0,n_samples,n_samples)
            
            bootstrap_X = X[indeces,:]
            bootstrap_y = y[indeces]
            
            model = self.classifier
            
            model.fit(bootstrap_X, bootstrap_y, attrs)
            
            self.ensemble.append(model)
            
        
    def _predict_one(self, X, attrs):
        pred = 0
        
        for i in range(self.n_classifiers):
            
            pred += self.ensemble[i]._predict_one(X, attrs)
            
            # Stop prediction procedure early when the current prediction is guaranteed
            # If the magnitude of the summed prediction is larger than the number of predictions remaining, then it is guaranteed to be its current sign
            # This process starts only after half the ensemble has made a prediction vote.
            if i >= (self.n_classifiers / 2):
                if abs(pred) > self.n_classifiers - i - 1:
                    break
        
        return int(math.copysign(1, pred))
        
    def predict(self, X, attrs):
        preds = []
        
        for i in range(len(X)):
            preds.append(self._predict_one(X[i], attrs))
        
        preds = np.array(preds)
        
        return preds