import pandas as pd
import numpy as np
import scipy as sp

cols = ['variance', 'skewness', 'kurtosis', 'entropy', 'label']
train = pd.read_csv(fr'data\train.csv', names=cols)
test = pd.read_csv(fr'data\test.csv', names=cols)
train['label'].replace(0, -1, inplace=True)
test['label'].replace(0, -1, inplace=True)

train.insert(0, 'bias', 1)
test.insert(0, 'bias', 1)

train_np = train.to_numpy()
test_np = test.to_numpy()

train_np_X = train.drop('label', axis=1).to_numpy()
train_np_Y = train['label'].to_numpy()

test_np_X = test.drop('label', axis=1).to_numpy()
test_np_Y = test['label'].to_numpy()

class NN:
    def __init__(self, num_features, width, rate_0=0.1, d = 1, init_w_normal = True):

        self.W = []
        self.width = width
        self.rate_0 = rate_0
        self.rate_t = rate_0
        self.t = 0
        self.d = d

        if init_w_normal:
            w_layer1 = np.random.randn(num_features, width + 1)
            w_layer2 = np.random.randn(width + 1, width + 1)
            w_layer3 = np.random.randn(width + 1, 1)
        else:
            w_layer1 = np.zeros((num_features, width + 1))
            w_layer2 = np.zeros((width + 1, width + 1))
            w_layer3 = np.zeros((width + 1, 1))

        self.W = [w_layer1, w_layer2, w_layer3]

    def sigmoid(self, x):
        return sp.special.expit(x)
        # return 1.0 / (1 + np.exp(-x))

    def sigmoid_gradient(self, x):
        return x * (1 - x)
    
    def fit(self, data, epochs=100):
        for epoch in np.arange(0, epochs):
            shuffled = np.copy(data)
            np.random.shuffle(shuffled)
            for row in shuffled:
                self.fit_one(row[0:-1], row[-1])
                
    def fit_one(self, x, y):
        A = [np.atleast_2d(x)]
        
        for layer in np.arange(0, len(self.W)):

            dotprod = A[layer].dot(self.W[layer])

            if layer < len(self.W) - 1:
                out = self.sigmoid(dotprod)
            else:
                out = dotprod
            A.append(out)

        partials = [A[-1] - y]
        
        for layer in np.arange(len(A) - 2, 0, -1):

            partial = partials[-1].dot(self.W[layer].T)
            partial = partial * self.sigmoid_gradient(A[layer])
            partials.append(partial)
        
        partials = partials[::-1]

        for layer in np.arange(0, len(self.W)):

            self.W[layer] += -self.rate_t * A[layer].T.dot(partials[layer])
            
        self.rate_t = self.rate_0 / (1 + (self.rate_0 * self.t / self.d))
        self.t += 1
        
    def predict(self, X):

        out = np.atleast_2d(X)

        for layer in np.arange(0, len(self.W)):
            if layer < len(self.W) - 1:
                out = self.sigmoid(np.dot(out, self.W[layer]))
            else:
                out = np.dot(out, self.W[layer])
        
        return out


for j in [5, 10, 25, 50, 100]:
    
    nn = NN(len(cols), j, .1, 2)

    nn.fit(train_np, 100)

    train_preds = []
    test_preds = []
    
    for i in train_np_X:
        pred = nn.predict(i)
        if pred > 0:
            train_preds.append(1)
        else:
            train_preds.append(-1)
    
    for i in test_np_X:
        pred = nn.predict(i)
        if pred > 0:
            test_preds.append(1)
        else:
            test_preds.append(-1)
        
    test_predictions = np.array(test_preds)
    train_predictions = np.array(train_preds)

    print(f"Neural Network with {j} width hidden layers, weights initialized w/ normal distribution.")
    print(f"Train Error: {np.round(1 - np.mean(train_predictions == train_np_Y), 4)}")
    print(f"Test Error: {np.round(1 - np.mean(test_predictions == test_np_Y), 4)}\n")
    
for j in [5, 10, 25, 50, 100]:

    nn = NN(len(cols), j, .1, 2, False)

    nn.fit(train_np, 100)

    train_preds = []
    test_preds = []
    
    for i in train_np_X:
        pred = nn.predict(i)
        if pred > 0:
            train_preds.append(1)
        else:
            train_preds.append(-1)
    
    for i in test_np_X:
        pred = nn.predict(i)
        if pred > 0:
            test_preds.append(1)
        else:
            test_preds.append(-1)
        
    test_predictions = np.array(test_preds)
    train_predictions = np.array(train_preds)

    print(f"Neural Network with {j} width hidden layers, weights initialized to zero.")
    print(f"Train Error: {np.round(1 - np.mean(train_predictions == train_np_Y), 4)}")
    print(f"Test Error: {np.round(1 - np.mean(test_predictions == test_np_Y), 4)}\n")